# Diversity Is All You Need

Authors: Rémy DESHAYES, Olivier DULCY

Final project for the Reinforcement Learning module at ENSAE Paris. Our project is based on the paper _Diversity Is All You Need_ by Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine (2018)

In the Reinforcement Learning (RL) paradigm one is interested in the study of agents and how they learn by trial and error. The fundamental idea is that rewarding or punishing an agent for its behavior makes it more likely to repeat it or discontinue it thereafter.
Section 1 of our report - please have a look at it - introduces the standard RL problem and spotlights some of its associated issues before introducing a workaround to the most pressing one. Section 2 delves into the workaround and explore a corresponding method that we later implement1, namely Diversity Is All You Need (DIAYN). Finally, section 3 briefly lay out where DIAYN stands in the RL literature before wrapping up the project with a few conclusive remarks

Our results are also available on video here : 
[pendulum_video](https://youtu.be/scjX7YhNthM), [mountain_car_video](https://youtu.be/XRDxTBMpc8g)

## Installation
### Build the Docker container
The code has been developped inside a Jupyter Lab Docker container allowing GPU access. The specific image is ``cschranz/gpu-jupyter:v1.3_cuda-10.1_ubuntu-18.04_python-only``. Please visit https://github.com/iot-salzburg/gpu-jupyter for further details.

You can build the container using the provided Dockerfile.

```bash
docker build . -t rl
```

The oneliner to run the container is the following one :
```bash
docker run -d --gpus all -p 8889:8888 -p 6006:6006 --name rl rl
```

Now, you have a container named ``rl`` running Jupyter Lab ! You can access it from ``http://127.0.0.1:8889``. Default password is ``gpu-jupyter``.

**Note :** The port 6006 is for ``tensorboard``. You can use ``tensorboard --logdir ./tensorboard --bind_all`` to run tensorboard.

Now, you can clone this repository :
```bash
git clone https://github.com/odulcy/DIAYN-ENSAE
```

### Miscellaneous
To run an arbitrary code in your container as a "normal" user, you can use :
```bash
docker exec -it rl bash
```

To get a root access, you can use :
```bash
docker exec -u root -it rl bash
```

### Optional dependencies

You can install ``pybulletgym`` for more environments. Please take a look at https://github.com/benelot/pybullet-gym#installing-pybullet-gym.

**Note :** you should run the commands as root inside the container to perform a system-wide installation.
