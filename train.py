import matplotlib.pyplot as plt
from IPython import display
import matplotlib.pyplot as plt
from make_train_step import generate_images
from make_generetor import my_generator
from make_train_step import train_step
generator = my_generator()

import time
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(2)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    if (step+1) % 10 == 0:
      print('.', end='', flush=True)