import gdown
import argparse
from zipfile import ZipFile

ids = {
    'linux': "1W3TTj-qSYehWqfXiN2Ye1PnRvfRXSRmG",
    'mac': "1rqcdKEUde1R_Od5WjT3xjdfmLfGRMOWf",
    'windows': "1N-oXV7Nm37pOlam3i0lvu-IPpQnRPVgp"
}

output = "blender_fracture_modifier_archive"

def main():
    parser = argparse.ArgumentParser(description="Downloads the Blender Fracture Modifier")
    parser.add_argument("--platform", type=str, help='Your platform name', required=True, choices=['linux', 'mac', 'windows'])
    args = parser.parse_args()

    gdown.download(id=ids[args.platform], output= output, quiet = False)

if __name__ == '__main__':
    main()
