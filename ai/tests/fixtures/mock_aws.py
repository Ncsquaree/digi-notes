from unittest.mock import MagicMock


class MockTextractClient:
    def detect_document_text(self, Document=None):
        # return a minimal Textract-like response
        return {
            'DocumentMetadata': {'Pages': 1},
            'Blocks': [
                {'BlockType': 'LINE', 'Text': 'This is a test line', 'Confidence': 98.5, 'Geometry': {'BoundingBox': {}}}
            ]
        }


class MockS3Client:
    def generate_presigned_url(self, ClientMethod, Params=None, ExpiresIn=3600):
        return f'https://s3.mock/{Params.get("Key")}'

    def get_object(self, Bucket, Key):
        return {'Body': b'fake', 'ContentLength': 4, 'ContentType': 'image/jpeg'}

    def download_file(self, Bucket, Key, Filename):
        # create a small placeholder file for tests; respect extension
        with open(Filename, 'wb') as fh:
            fh.write(b'\xff\xd8\xff\xdb')
        return True


def fake_boto3_client(name, *a, **kw):
    if name == 'textract':
        return MockTextractClient()
    if name == 's3':
        return MockS3Client()
    return MagicMock()
