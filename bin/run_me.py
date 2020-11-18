import data.data_reader as dr
import data.config_reader as cr
import matplotlib.pyplot as plt
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model generator')
    parser.add_argument('--model_path', type=str, help='Use: ../models/th_label/ or ../models/local_extrema/', default='../models/th_label/')
    args = parser.parse_args()

model_path = args.model_path

config = cr.config_reader(model_path=model_path)
data = dr.data_reader(config)

fig, axs = plt.subplots(4, sharex=True)
fig.suptitle('Label visualisation - thresholds')
axs[0].set_ylabel("Prices")
axs[0].plot(data.get_prices_test(),'g')
axs[1].set_ylabel("Labels - continuous")
axs[1].plot(data.get_coef_array_test(),'brown')
axs[2].set_ylabel("Labels - discrete")
axs[2].set_xlabel("time [minutes]")
axs[2].plot(np.where(data.get_labels_test() > 1.5)[0], data.get_labels_test()[np.where(data.get_labels_test() > 1.5)[0]],'bo')
axs[2].plot(np.where(data.get_labels_test() == 1)[0], data.get_labels_test()[np.where(data.get_labels_test() == 1)[0]],'go')
axs[2].plot(np.where(data.get_labels_test() < 0.5)[0], data.get_labels_test()[np.where(data.get_labels_test() < 0.5)[0]],'ro')
axs[3].set_ylabel("Loss scaler")
axs[3].plot(data.get_loss_scaler_test(),'black')

plt.show()
