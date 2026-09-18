import os
import cv2
import numpy as np
import logging
import traceback
import requests

# Lightweight camera processing that avoids importing heavy ML libs at import time.
# Local YOLO usage is only enabled when USE_LOCAL_MODEL=1.
USE_LOCAL = os.environ.get("USE_LOCAL_MODEL", "0") == "1"

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# 'connect' provides visual search and web search
try:
    # Preferred: package import when running as package
    from Excel.Server import connect
except Exception:
    try:
        # Relative import when running as package/module
        from . import connect
    except Exception:
        # Fallback: plain import when running as a flat module
        import connect


def process_image_bytes(image_bytes):
    """Decode image bytes to OpenCV BGR image and return it."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# Heuristic keyboard detector using contour analysis
def detect_keyboard(frame):
    """Return (match:bool, confidence:float) if the frame likely contains a keyboard.

    Strategy: detect many small rectangular key-shaped contours arranged in rows.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downscale for speed
        h, w = gray.shape[:2]
        scale = 600.0 / max(h, w) if max(h, w) > 600 else 1.0
        if scale != 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # Adaptive threshold to highlight keys
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
        # Morphology to join key areas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours that look like keys
        key_like = 0
        areas = []
        H, W = morph.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50 or area > (W * H * 0.2):
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = cw / float(max(ch, 1))
            # keys are often roughly rectangular with moderate aspect ratio
            if 0.4 <= ar <= 3.0:
                # exclude very thin contours
                if cw > 8 and ch > 6:
                    key_like += 1
                    areas.append(area)
        # Heuristic: keyboards have many small key-like contours
        if key_like >= 20:
            # confidence scaled to number of keys
            conf = min(0.95, 0.02 * key_like)
            return True, round(conf, 2)
        return False, 0.0
    except Exception:
        logging.exception('detect_keyboard failed')
        return False, 0.0


# Template logo matching (simple normalized cross-correlation)
def match_logo(frame, template_path='templates/hp_logo.png'):
    """Return (match:bool, confidence:float) if template found in frame.

    Requires a small template image at templates/hp_logo.png. If not present, returns (False,0).
    """
    try:
        if not os.path.exists(template_path):
            return False, 0.0
        tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            return False, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize template if larger than frame
        th, tw = tpl.shape[:2]
        fh, fw = gray.shape[:2]
        if th > fh or tw > fw:
            # scale down template
            scale = min(fh / th, fw / tw) * 0.8
            tpl = cv2.resize(tpl, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA)
            th, tw = tpl.shape[:2]
        # Template matching
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        # Confidence threshold
        if max_val >= 0.6:
            return True, float(max_val)
        return False, float(max_val)
    except Exception:
        logging.exception('match_logo failed')
        return False, 0.0


def process_image_search_only(frame):
    """Perform visual-search + OCR on the frame and return results without local model."""
    results = {"detections": []}
    img_h, img_w = frame.shape[:2]
    # Try full-image visual search
    try:
        _, jpg = cv2.imencode('.jpg', frame)
        s_full = connect.search(query=None, image_bytes=jpg.tobytes(), top_k=5)
        if s_full.get('success') and s_full.get('results'):
            results['detections'].append({'source': 'full_image_visual', 'web_results': s_full.get('results')})
            return results
    except Exception:
        logging.exception('full-image visual search failed')

    # Try a couple of crops
    crops = []
    sz = min(160, max(64, min(img_h, img_w)))
    cy, cx = img_h // 2, img_w // 2
    y1 = max(0, cy - sz // 2); x1 = max(0, cx - sz // 2)
    crops.append(frame[y1:y1+sz, x1:x1+sz])
    qh, qw = max(10, img_h//4), max(10, img_w//4)
    crops.append(frame[0:qh, 0:qw])

    for c in crops:
        try:
            _, cj = cv2.imencode('.jpg', c)
            sres = connect.search(query=None, image_bytes=cj.tobytes(), top_k=5)
            if sres.get('success') and sres.get('results'):
                results['detections'].append({'source': 'crop_visual', 'web_results': sres.get('results')})
                return results
        except Exception:
            logging.exception('crop visual search failed')

    # OCR fallback
    try:
        if pytesseract is not None and Image is not None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil, config='--psm 6').strip()
            if text:
                stext = connect.search(query=text, image_bytes=None, top_k=5)
                if stext.get('success') and stext.get('results'):
                    results['detections'].append({'source': 'ocr_text', 'ocr': text, 'web_results': stext.get('results')})
                    return results
    except Exception:
        logging.exception('ocr fallback failed')

    return results


def process_image_with_local_model(frame):
    """Run local YOLO if enabled. This imports ultralytics lazily and is only used when USE_LOCAL==True."""
    if not USE_LOCAL:
        raise RuntimeError('Local model disabled')
    # Lazy import of ultralytics to avoid import at module load
    from ultralytics import YOLO
    model_name = os.environ.get('YOLO_MODEL', 'yolo8n.pt')
    model = YOLO(model_name)
    # small imgsz
    imgsz = int(os.environ.get('IMG_SIZE', '112'))
    results = model(frame, imgsz=imgsz, device='cpu')
    detections = []
    for res in results:
        for box in res.boxes:
            try:
                class_id = int(box.cls[0])
            except Exception:
                class_id = None
            try:
                conf = float(box.conf[0])
            except Exception:
                conf = None
            name = res.names.get(class_id, str(class_id)) if class_id is not None else ''
            detections.append({'class': name, 'confidence': conf})
    return {'detections': detections}


def process_image(image_bytes):
    """Main entry used by server: given image bytes return detection/search results."""
    frame = process_image_bytes(image_bytes)
    if frame is None:
        return {'success': False, 'message': 'Could not decode image'}

    # First run lightweight heuristics (logo + keyboard detector) on multiple rotations
    debug_basic = {'rotations': []}
    logo_found = False
    kb_found = False
    logo_best = (False, 0.0, 0)  # match, conf, rotation
    kb_best = (False, 0.0, 0)
    # try 0, 90, 270 degrees to handle rotated photos
    rotations = [0, 90, 270]
    for rot in rotations:
        if rot == 0:
            frm = frame
        else:
            # rotate clockwise
            if rot == 90:
                frm = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frm = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        try:
            logo_match, logo_conf = match_logo(frm, template_path=os.path.join(os.path.dirname(__file__), 'templates', 'hp_logo.png'))
        except Exception:
            logo_match, logo_conf = False, 0.0
        try:
            kb_match, kb_conf = detect_keyboard(frm)
        except Exception:
            kb_match, kb_conf = False, 0.0
        debug_basic['rotations'].append({'rotation': rot, 'logo_match': logo_match, 'logo_conf': logo_conf, 'kb_match': kb_match, 'kb_conf': kb_conf})
        if logo_match and logo_conf > logo_best[1]:
            logo_best = (True, logo_conf, rot)
        if kb_match and kb_conf > kb_best[1]:
            kb_best = (True, kb_conf, rot)

    if logo_best[0]:
        return {'success': True, 'data': {'source': 'heuristic', 'label': 'HP keyboard', 'confidence': logo_best[1], 'rotation': logo_best[2]}, 'debug': {'heuristic': debug_basic}}
    if kb_best[0]:
        return {'success': True, 'data': {'source': 'heuristic', 'label': 'keyboard', 'confidence': kb_best[1], 'rotation': kb_best[2]}, 'debug': {'heuristic': debug_basic}}

    if USE_LOCAL:
        try:
            out = process_image_with_local_model(frame)
            return {'success': True, 'data': out, 'debug': {'note': 'used_local_model', 'heuristic': debug_basic}}
        except Exception:
            logging.exception('local model processing failed, falling back to search-only')
            # fall through to search

    # Perform visual-search + OCR with detailed debug info
    debug = {
        'heuristic': debug_basic,
        'full_image': None,
        'crops': [],
        'ocr_text': None,
        'exceptions': []
    }
    try:
        # Full-image visual search
        try:
            _, jpg = cv2.imencode('.jpg', frame)
            s_full = connect.search(query=None, image_bytes=jpg.tobytes(), top_k=5)
            ok = bool(s_full.get('success') and s_full.get('results'))
            count = len(s_full.get('results') or [])
            debug['full_image'] = {'success': ok, 'count': count}
            if ok:
                return {'success': True, 'data': {'source': 'full_image_visual', 'results': s_full.get('results')}, 'debug': debug}
        except Exception:
            tb = traceback.format_exc()
            logging.exception('full-image visual search failed')
            debug['exceptions'].append({'stage': 'full_image', 'trace': tb})

        # Crops
        crops = []
        h, w = frame.shape[:2]
        # center and quarter crops
        for size in (min(160, max(64, min(h, w))), min(128, max(64, min(h, w)))):
            cy, cx = h // 2, w // 2
            y1 = max(0, cy - size // 2); x1 = max(0, cx - size // 2)
            crops.append(frame[y1:y1+size, x1:x1+size])
        qh, qw = max(10, h//4), max(10, w//4)
        crops.append(frame[0:qh, 0:qw])

        for c in crops:
            try:
                _, cj = cv2.imencode('.jpg', c)
                sres = connect.search(query=None, image_bytes=cj.tobytes(), top_k=5)
                ok = bool(sres.get('success') and sres.get('results'))
                count = len(sres.get('results') or [])
                debug['crops'].append({'size': c.shape[:2], 'success': ok, 'count': count})
                if ok:
                    return {'success': True, 'data': {'source': 'crop_visual', 'results': sres.get('results')}, 'debug': debug}
            except Exception:
                tb = traceback.format_exc()
                logging.exception('crop visual search failed')
                debug['exceptions'].append({'stage': 'crop', 'trace': tb})

        # OCR fallback on the whole image, prefer pytesseract but fall back to OCR.space if available
        try:
            text = ""
            # prepare jpg bytes for remote OCR if needed
            try:
                _, full_jpg = cv2.imencode('.jpg', frame)
                full_jpg_bytes = full_jpg.tobytes()
            except Exception:
                full_jpg_bytes = None

            # Try pytesseract first if available
            if pytesseract is not None and Image is not None:
                try:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(pil, config='--psm 6').strip()
                except Exception as e:
                    # If Tesseract binary missing or other error, fall back to remote OCR
                    tb = traceback.format_exc()
                    logging.exception('pytesseract failed')
                    debug['exceptions'].append({'stage': 'pytesseract', 'trace': tb})

            # If no local text, try OCR.Space when API key is provided
            if (not text) and full_jpg_bytes is not None:
                ocr_key = os.environ.get('OCR_SPACE_API_KEY')
                if ocr_key:
                    try:
                        def ocr_space_api(image_bytes, api_key, language='eng'):
                            url = 'https://api.ocr.space/parse/image'
                            files = {'file': ('image.jpg', image_bytes)}
                            data = {'apikey': api_key, 'language': language, 'isOverlayRequired': False}
                            r = requests.post(url, files=files, data=data, timeout=30)
                            r.raise_for_status()
                            j = r.json()
                            parsed = j.get('ParsedResults')
                            if parsed and len(parsed) > 0:
                                return parsed[0].get('ParsedText', '').strip()
                            return ''

                        text = ocr_space_api(full_jpg_bytes, ocr_key)
                        debug['ocr_used'] = 'ocr_space'
                    except Exception:
                        tb = traceback.format_exc()
                        logging.exception('OCR.space fallback failed')
                        debug['exceptions'].append({'stage': 'ocr_space', 'trace': tb})

            debug['ocr_text'] = text
            if text:
                stext = connect.search(query=text, image_bytes=None, top_k=5)
                ok = bool(stext.get('success') and stext.get('results'))
                count = len(stext.get('results') or [])
                debug['ocr_search'] = {'success': ok, 'count': count}
                if ok:
                    return {'success': True, 'data': {'source': 'ocr_text', 'ocr': text, 'results': stext.get('results')}, 'debug': debug}
        except Exception:
            tb = traceback.format_exc()
            logging.exception('OCR/text search failed')
            debug['exceptions'].append({'stage': 'ocr', 'trace': tb})

        # Nothing matched
        return {'success': False, 'message': 'no_match', 'debug': debug}

    except Exception:
        tb = traceback.format_exc()
        logging.exception('unexpected error in process_image')
        return {'success': False, 'message': 'error', 'debug': {'exceptions': [{'stage': 'unexpected', 'trace': tb}]}}
