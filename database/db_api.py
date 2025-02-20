from flask import Flask, request, jsonify
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
    embedding = np.frombuffer(embedding_bytes, dtype=np.float64).reshape(256, 1)
    db.add_user(username, embedding)
    return jsonify({"status": "success"})

@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    embedding = db.get_user(username)
    if embedding is not None:
        # Encode embedding as base64
        embedding_bytes = embedding.tobytes()
        embedding_b64 = base64.b64encode(embedding_bytes).decode()
        return jsonify({"embedding": embedding_b64})
    return jsonify({"error": "User not found"}), 404

@app.route('/users', methods=['GET'])
def get_all_users():
    embeddings = db.get_all_embeddings()
    # Convert embeddings to base64
    return jsonify({
        username: base64.b64encode(emb.tobytes()).decode()
        for username, emb in embeddings.items()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
