access = input("Give access to camera: ")

while access.lower() not in ["yes", "no"]:
    print("Invalid input. Please enter 'yes' or 'no'.")
    access = input("Give access to camera: ")

if access.lower() == "yes":
    print("Access granted to camera.")
    from camera import start_camera
    start_camera()
elif access.lower() == "no":
    print("Access denied to camera. \nCant continue...")
