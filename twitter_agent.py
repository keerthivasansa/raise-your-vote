import tweepy
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class TwitterAgent:
    def __init__(self, label_to_twitter):
        self.label_to_twitter = label_to_twitter
        self.accounts = []
        
        # Initialize multiple Twitter accounts (up to 10) for demonstration
        for i in range(10):
            consumer_key = os.getenv(f'TWITTER_API_KEY_{i+1}', os.getenv('TWITTER_API_KEY') if i == 0 else None)
            consumer_secret = os.getenv(f'TWITTER_API_SECRET_{i+1}', os.getenv('TWITTER_API_SECRET') if i == 0 else None)
            access_token = os.getenv(f'TWITTER_ACCESS_TOKEN_{i+1}', os.getenv('TWITTER_ACCESS_TOKEN') if i == 0 else None)
            access_token_secret = os.getenv(f'TWITTER_ACCESS_TOKEN_SECRET_{i+1}', os.getenv('TWITTER_ACCESS_TOKEN_SECRET') if i == 0 else None)
            
            if all([consumer_key, consumer_secret, access_token, access_token_secret]):
                try:
                    client = tweepy.Client(
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret
                    )
                    auth = tweepy.OAuth1UserHandler(
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret
                    )
                    api = tweepy.API(auth)
                    user = client.get_me()
                    self.accounts.append({
                        'client': client,
                        'api': api,
                        'username': user.data.username
                    })
                    logger.info(f"Initialized Twitter account {i+1}: {user.data.username}")
                except Exception as e:
                    logger.error(f"Failed to initialize Twitter account {i+1}: {str(e)}")
            else:
                logger.warning(f"Skipping Twitter account {i+1}: Missing credentials")
        
        if not self.accounts:
            logger.error("No valid Twitter accounts initialized")
            raise Exception("No valid Twitter accounts initialized")
        
        logger.info(f"Initialized {len(self.accounts)} Twitter account(s)")

    def select_account(self):
        """Select a random Twitter account for posting"""
        if not self.accounts:
            logger.error("No Twitter accounts available")
            raise Exception("No Twitter accounts available")
        account = random.choice(self.accounts)
        logger.info(f"Selected Twitter account: {account['username']}")
        return account

    def post_complaint(self, complaint):
        try:
            account = self.select_account()
            client = account['client']
            api = account['api']
            
            # Use complaint_text instead of text
            message = f"{complaint.get('complaint_text', 'No complaint text provided')} in {complaint.get('state', 'India')}. @Yuvi564321"
            twitter_handle = complaint.get('twitter_handle', '@Yuvi564321')
            if twitter_handle != '@Yuvi564321':
                message += f" {twitter_handle}"
            
            hashtags = complaint.get('hashtags', [])
            if hashtags:
                message += f" {' '.join(hashtags)}"
            
            if len(message) > 280:
                message = message[:277] + "..."
            
            logger.info(f"Attempting to post tweet from {account['username']}: {message}")

            media_ids = []
            image_path = complaint.get('image_path')
            logger.info('sending complaint: ', complaint)
            if image_path and os.path.exists(image_path):
                try:
                    media = api.media_upload(image_path)
                    media_ids.append(media.media_id)
                    logger.info(f"Uploaded media: {image_path}, media_id: {media.media_id}")
                except Exception as e:
                    logger.error(f"Media upload error for {image_path}: {str(e)}")
                    media_ids = []
            
            try:
                response = client.create_tweet(text=message, media_ids=media_ids if media_ids else None)
                logger.info(f"Tweet posted successfully from {account['username']}: {response.data['id']}")
                return {'success': True, 'tweet_id': response.data['id'], 'username': account['username']}
            except Exception as e:
                logger.error(f"Tweet posting error from {account['username']}: {str(e)}")
                return {'success': False, 'error': f"Tweet posting failed: {str(e)}"}
        
        except Exception as e:
            logger.error(f"General error in post_complaint: {str(e)}")
            return {'success': False, 'error': f"Failed to post complaint: {str(e)}"}

    def test_tweet(self):
        """Post a test tweet from a randomly selected account"""
        try:
            account = self.select_account()
            client = account['client']
            message = f"Test tweet from VoiceForAll @Yuvi564321 at {datetime.now().isoformat()}"
            logger.info(f"Attempting test tweet from {account['username']}: {message}")
            response = client.create_tweet(text=message)
            logger.info(f"Test tweet posted successfully from {account['username']}: {response.data['id']}")
            return {'success': True, 'tweet_id': response.data['id'], 'username': account['username']}
        except Exception as e:
            logger.error(f"Test tweet error: {str(e)}")
            return {'success': False, 'error': f"Test tweet failed: {str(e)}"}