import firebase_admin
from firebase_admin import credentials, firestore

firebase_creds = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(firebase_creds)

firebase_admin.initialize_app(cred)

db = firestore.client()
