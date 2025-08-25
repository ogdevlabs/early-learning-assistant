import sounddevice as sd
import json
import os

def list_devices():
    devices = sd.query_devices()
    print("Available audio devices:")
    for idx, dev in enumerate(devices):
        dev_type = []
        if dev['max_input_channels'] > 0:
            dev_type.append('Input')
        if dev['max_output_channels'] > 0:
            dev_type.append('Output')
        print(f"[{idx}] {dev['name']} - {', '.join(dev_type)}")
    return devices

def select_device(devices, device_type):
    while True:
        try:
            idx = int(input(f"Enter the index of the {device_type} device to use: "))
            if 0 <= idx < len(devices):
                if device_type == 'input' and devices[idx]['max_input_channels'] > 0:
                    return idx
                elif device_type == 'output' and devices[idx]['max_output_channels'] > 0:
                    return idx
                else:
                    print(f"Device [{idx}] is not a valid {device_type} device.")
            else:
                print("Invalid index.")
        except ValueError:
            print("Please enter a valid integer index.")

def save_config(input_idx, output_idx, config_path):
    config = {
        'input_device': input_idx,
        'output_device': output_idx
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Configuration saved to {config_path}")

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'audio_device_config.json')
    devices = list_devices()
    input_idx = select_device(devices, 'input')
    output_idx = select_device(devices, 'output')
    save_config(input_idx, output_idx, config_path)

if __name__ == "__main__":
    main()
