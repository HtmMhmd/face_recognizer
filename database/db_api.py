from flask import Flask, request, jsonify, render_template
from db_handler import FaceDatabase
import numpy as np
import base64

app = Flask(__name__)
db = FaceDatabase()

@app.route('/user', methods=['POST'])
def add_user():
    data = request.json
    username = data['username']
    # Decode base64 embedding
    embedding_bytes = base64.b64decode(data['embedding'])
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(512,1)
    db.add_user(username, embedding)
    return jsonify({"status": "success"})

@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    try:
        user_data = db.get_user(username)
        embedding = user_data['embedding']
        date_added = user_data['date_added']
        last_login = user_data['last_login']
        # Encode embedding as base64
        embedding_bytes = embedding.tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode()
        return jsonify({
            "embedding": embedding_b64,
            "date_added": date_added,
            "last_login": last_login
        })
    except ValueError:
        return jsonify({"error": "User not found"}), 404

@app.route('/users', methods=['GET'])
def get_all_users():
    embeddings = db.get_all_embeddings()
    # Convert embeddings to base64
    users = {
        username: {
            "embedding": base64.b64encode(emb['embedding'].tobytes()).decode(),
            "date_added": emb['date_added'],
            "last_login": emb['last_login']
        }
        for username, emb in embeddings.items()
    }
    return jsonify(users)

@app.route('/database')
def database():
    embeddings = db.get_all_embeddings()
    users = [
        {
            "username": username,
            "date_added": emb['date_added'],
            "last_login": emb['last_login']
        }
        for username, emb in embeddings.items()
    ]
    return render_template('database.html', users=users)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
