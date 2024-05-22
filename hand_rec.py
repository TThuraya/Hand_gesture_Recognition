from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)




def launch_script(script_name):
  try:
    process = subprocess.Popen(["python3", script_name])
    return jsonify({"message": f"Launched script: {script_name}"})
  except Exception as e:
    return jsonify({"error": f"Error launching script: {e}"}), 500


@app.route("/launch/gesture", methods=["POST"])
def launch_gesture():
  return launch_script("gesture_rec.py")

@app.route("/launch/face_mesh", methods=["POST"])
def launch_face_mesh():
  return launch_script("face_mesh.py")


if __name__ == '__main__':
   app.run(debug=True, port=8080)