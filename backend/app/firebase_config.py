from firebase_admin import credentials, firestore
import os
import json
import firebase_admin
from firebase_admin import credentials

if os.path.exists("app/firebase_key.json"):
    cred = credentials.Certificate("app/firebase_key.json")
else:
    firebase_creds = json.loads(
        os.environ["FIREBASE_CREDENTIALS"]
    )
    cred = credentials.Certificate(firebase_creds)

firebase_admin.initialize_app(cred)
db = firestore.client()
