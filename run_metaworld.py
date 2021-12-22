import metaworld
import random
import matplotlib.pyplot as plt
import cv2

training_envs = []
images = []
offscreen = True
record_vid = False
steps_per_env = 50


#ML1
# ml1 = metaworld.ML1('assembly-v2')
# env = ml1.train_classes['assembly-v2']()
# task = random.choice(ml1.train_tasks)
# env.set_task(task)
# training_envs.append(env)

#ML10
ml10 = metaworld.ML10()
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

for env in training_envs:
  obs = env.reset()
  for i in range(steps_per_env):
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)
    
    img = env.render(offscreen=offscreen, camera_name='behindGripper')
    if offscreen:
      img = cv2.flip(img, 0)
      images.append(img)
      cv2.imshow("R", img)
      cv2.waitKey(5)

if record_vid:
  size = images[0].shape[:2]
  out = cv2.VideoWriter('/home/shivin/Desktop/run.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (size[1], size[0]))
  for img in images:
    out.write(img)
  out.release()