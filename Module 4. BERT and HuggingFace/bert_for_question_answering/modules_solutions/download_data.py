import os
import wget
import argparse


def download_squad_dataset(path: str):
    """
    Downloads the SQuAD v1.1 dataset (consisting of three files) to the specified path.
    :param path: the path to download the files to.
    :return: None
    """
    assert os.path.exists(path), f"The specified path does not exist: {path}."
    files = [
        ('train-v1.1.json', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'),
        ('dev-v1.1.json', 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'),
        ('evaluate-v1.1.py', 'https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py')
    ]
    for filename, url_link in files:
        wget.download(url_link, os.path.join(path, filename))


if __name__ == "__main__":
    """
    Some combinations of some versions of MacOS and python3 may raise a urllib.error.URLError or an 
    ssl.SSLCertVerificationError stating "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed".
    To solve this, set this environment variables in your terminal before calling this script:
    CERT_PATH=$(python -m certifi)
    SSL_CERT_FILE=${CERT_PATH}
    REQUESTS_CA_BUNDLE=${CERT_PATH}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The folder to download the train, validation and test datasets.")
    args = parser.parse_args()
    download_squad_dataset(args.path)
