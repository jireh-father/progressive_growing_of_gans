from flask import Flask, render_template, request
import util_scripts
import uuid
import os
import numpy as np
import config
import tfutil
import misc
import time

app = Flask(__name__)

snapshot = 5409
snapshot = None
run_id = 2
latent_cnt = 512
img_size = 512
step = 0.1
min = -5.0
max = 5.0

misc.init_output_logging()
np.random.seed(config.random_seed)
print('Initializing TensorFlow...')
os.environ.update(config.env)
tfutil.init_tf(config.tf_config)
print('Running %s()...' % config.train['func'])
network_pkl = misc.locate_network_pkl(run_id)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(run_id, snapshot)
if not os.path.isdir("static/generated"):
    os.makedirs("static/generated")


@app.route("/", methods=['GET', 'POST'])
def main_page():
    file_name = "%s.png" % uuid.uuid4()
    path = os.path.join(os.path.dirname(__file__), "static/generated", file_name)

    if 'is_random' not in request.args or request.args["is_random"] == "1":
        np.random.seed(int(time.time()))
        latents = np.random.randn(1, latent_cnt).astype(np.float32)
    else:
        latent_list = []
        for i in range(latent_cnt):
            latent_list.append(float(request.args['latent_%d' % i]))
        latents = np.array([latent_list]).astype(np.float32)
        print(latents)

    util_scripts.generate_fake_images(run_id, path=path, latent=latents, Gs=Gs)
    latent = latents[0].round(1)
    # min = latent.min()
    # max = latent.max()
    return render_template("main.html", file_name=file_name, min=min, max=max, latent=list(latent),
                           step=step, latent_cnt=latent_cnt, img_size=img_size)


@app.route("/calc", methods=['GET', 'POST'])
def latent_vector_calc():
    first_file_name = "%s.png" % uuid.uuid4()
    second_file_name = "%s.png" % uuid.uuid4()
    result_file_name = "%s.png" % uuid.uuid4()
    first_path = os.path.join(os.path.dirname(__file__), "static/generated", first_file_name)
    second_path = os.path.join(os.path.dirname(__file__), "static/generated", second_file_name)
    result_path = os.path.join(os.path.dirname(__file__), "static/generated", result_file_name)

    if 'first_is_random' not in request.args or request.args["first_is_random"] == "1":
        np.random.seed(int(time.time()))
        first_latents = np.random.randn(1, latent_cnt).astype(np.float32)
    else:
        latent_list = []
        for i in range(latent_cnt):
            latent_list.append(float(request.args['first_latent_%d' % i]))
        first_latents = np.array([latent_list]).astype(np.float32)

    if 'second_is_random' not in request.args or request.args["second_is_random"] == "1":
        np.random.seed(int(time.time()) * 2)
        second_latents = np.random.randn(1, latent_cnt).astype(np.float32)
    else:
        latent_list = []
        for i in range(latent_cnt):
            latent_list.append(float(request.args['second_latent_%d' % i]))
        second_latents = np.array([latent_list]).astype(np.float32)
    util_scripts.generate_fake_images(run_id, path=first_path, latent=first_latents, Gs=Gs)
    util_scripts.generate_fake_images(run_id, path=second_path, latent=second_latents, Gs=Gs)
    if 'operation' not in request.args or request.args["operation"] == "+":
        operation = "+"
        result_latent = first_latents + second_latents
    else:
        operation = "_"
        result_latent = first_latents - second_latents
    util_scripts.generate_fake_images(run_id, path=result_path, latent=result_latent, Gs=Gs)
    first_latent = first_latents[0].round(1)
    second_latent = second_latents[0].round(1)
    # min = latent.min()
    # max = latent.max()
    return render_template("latent_vector_calc.html", first_file_name=first_file_name,
                           second_file_name=second_file_name, result_file_name=result_file_name,
                           min=min, max=max, first_latent=list(first_latent), second_latent=list(second_latent),
                           step=step, latent_cnt=latent_cnt, img_size=img_size, operation=operation)
