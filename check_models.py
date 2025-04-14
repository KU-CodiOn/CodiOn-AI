import os
import boto3
from botocore.exceptions import ClientError

def check_models():
    # AWS credentials should be set in environment variables
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    s3 = boto3.client('s3')
    
    try:
        # List objects in the bucket
        response = s3.list_objects_v2(
            Bucket='your-bucket-name',
            Prefix='models/'
        )
        
        if 'Contents' in response:
            print("Available models:")
            for obj in response['Contents']:
                print(f"- {obj['Key']}")
        else:
            print("No models found in the bucket.")
            
    except ClientError as e:
        print(f"Error accessing S3: {e}")

if __name__ == "__main__":
    check_models() 