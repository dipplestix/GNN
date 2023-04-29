import os
import requests
import tarfile


# Adapted from https://github.com/amazon-science/gcp-with-gnns-example/blob/main/gc_example.ipynb with chatGPT
def download_file(url, output_directory):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Save the content to a file
        output_file = os.path.join(output_directory, 'instances.tar')
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"File downloaded to {output_file}")
        return output_file
    else:
        print(f"Error {response.status_code}: Unable to download file.")
        return None


def extract_tar_file(tar_file_path, extract_path):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=extract_path)
        print(f"File extracted to {extract_path}")


def main():
    input_data_path = './data/input/COLOR/instances'
    url = 'https://mat.tepper.cmu.edu/COLOR/instances/instances.tar'

    # Create the input_data_path directory if it doesn't exist
    if not os.path.exists(input_data_path):
        os.makedirs(input_data_path)

    # Download the file
    tar_file_path = download_file(url, input_data_path)

    # Extract the tar file
    if tar_file_path is not None:
        extract_tar_file(tar_file_path, input_data_path)


if __name__ == '__main__':
    main()
