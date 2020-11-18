import numpy as np
import pandas as pd
import copy
from typing import Tuple



def make_training_data(boundary: float,
					   coef_array_sum: np.array) -> Tuple[np.array, np.array]:
	"""Make all training data (can be used for producing testing perfect labels).
	labels - vector contains all labels (buy - 0, do nothing - 1, sell - 2)
	loss_scaler - vector contains weigths for scaling the loss NN loss function (we tries to eliminate huge difference
					in labels distribution, 'do nothing' label appears a lot more often than others."""
	labels = np.zeros(len(coef_array_sum),dtype=int)
	loss_scaler = get_buy_scale(coef_array_sum=coef_array_sum,
								boundary=boundary)

	# Creates vector with perfect labels based on coefArraySum and the boundary
	# buy   |   do nothing  |   sell
	#   -boundary		boundary
	for i in range(0, len(coef_array_sum)):
		if coef_array_sum[i] > boundary:
			labels[i] = 2  # sell
		elif coef_array_sum[i] < -boundary:
			labels[i] = 0  # buy
		else:
			labels[i] = 1  # no action

	return labels, loss_scaler

def get_up_down_peaks(prices: np.array,
               up_limit: float,
               down_limit: float,
               n_limits: int,
               window: int) -> np.array:
    prices = prices.copy()
    prices = get_smoothed_prices(prices, window=window)
    sol = np.zeros((len(prices), 2))
    sol = perfect_buy_sell_finder(prices, sol, up_limit, down_limit, n_limits)
    peaks = np.ones(len(prices), dtype=int)
    index = np.where((sol[:, 0] + sol[:, 1]) > 0)[0]
    peaks[sol[index, 0].astype(int)] = 0
    peaks[sol[index, 1].astype(int)] = 2

    return peaks

def get_coef_array_up_down(prices: np.array,
                    coefficient_power: int,
                    window: int,
                    n_limits: int,
                    up_limit: float,
                    down_limit: float) -> np.array:

    up_down_peaks = get_up_down_peaks(prices=prices,
               up_limit=up_limit,
               down_limit=down_limit,
               n_limits=n_limits,
               window=window)

    smoothed_prices = get_smoothed_prices(prices, window=window)
    coef_array_sell: np.array = get_coef_array_sell(peaks=up_down_peaks,
                                                    prices=smoothed_prices,
                                                    coefficient_power=coefficient_power)
    coef_array_buy: np.array = get_coef_array_buy(peaks=up_down_peaks,
                                                  prices=smoothed_prices,
                                                  coefficient_power=coefficient_power)

    # coefArraySum definition
    coef_array_sum = coef_array_sell - coef_array_buy  # x^coefficient_power - y^coefficient_power shape
    # normalise
    coef_array_sum = np.where(coef_array_sum >= 1, 1, coef_array_sum)
    coef_array_sum = np.where(coef_array_sum <= -1, -1, coef_array_sum)

    return coef_array_sum


def get_coef_array_threshold(prices: np.array,
                                 distance: int,
                                 buy_threshold: float,
                                 sell_threshold: float,
                                 coefficient_power: int,
                                 window: int,
                                 normalise: bool) -> np.array:
    price = get_smoothed_prices(prices, window=window)
    temp = pd.DataFrame()
    for i in range(distance+1):
        if i == distance:
            temp[i] = -price[i:]
        elif i == 0:
            temp[i] = price[i:-(distance - i)] * (distance*(distance+1)/2)
        else:
            temp[i] = price[i:-(distance - i)] * (-(distance - i + 1))
    temp_ = (temp.sum(axis=1) / price[:-distance]).values

    if normalise:
        temp_ = np.concatenate([temp_, np.zeros(len(price) - len(temp))])
        stop_loss_coef = np.percentile(np.abs(temp_), 99, interpolation='lower')

        temp_ = temp_/stop_loss_coef
        temp_ = np.where(temp_ >= 1, 1, temp_)
        temp_ = np.where(temp_ <= -1, -1, temp_)
        temp_ = temp_**coefficient_power

    return temp_


def get_smoothed_prices(prices: np.array,
                        window: int) -> np.array:
    """the window should be an odd number"""
    if window == 1 or window == 0:
        return prices
    else:
        prices = prices.copy()
        temp = np.cumsum(prices)
        temp = temp[window:] - temp[:-window]
        prices[window // 2 + 1: -(window // 2)] = temp / window
        return prices


def perfect_buy_sell_finder(price: np.array,
                            sol: np.array,
                            limit_p: float,
                            limit_2p: float,
                            n_limits: int) -> np.array:
    # calculate limits
    A = pd.DataFrame({'limit': price})
    limit = list(A.rolling(n_limits, min_periods=1).mean()['limit'] * limit_p / 100.)
    limit2 = list(A.rolling(n_limits, min_periods=1).mean()['limit'] * limit_2p / 100.)
    n = len(price)
    count = 0
    i = 0
    flag = 0
    while i < n - 1:
        while (i < n - 1) and (price[i + 1] <= price[i]):
            i += 1
        if i == n - 1:
            break
        if (price[i] <= price[int(sol[count][0])]) or flag == 1:
            sol[count][0] = i
            flag = 0
        i += 1
        while (i < n) and (price[i] > price[i - 1]):
            i += 1
        sol[count][1] = i - 1
        if i == n:
            break
        # if small profit - do nothing
        if (price[int(sol[count][1])] - price[int(sol[count][0])]) > limit[i]:
            # special code
            if count > 0 and (price[int(sol[count - 1][1])] - price[int(sol[count][0])]) < limit2[i]:
                if count > 1 and ((price[int(sol[count][1])] - price[int(sol[count - 1][1])]) > 0):
                    sol[count - 1][1] = sol[count][1]
                sol[count][1] = 0
                sol[count][0] = 0
                flag = 1
            else:
                count = count + 1
                flag = 1
        else:
            if count > 0 and (price[int(sol[count][1])] - price[int(sol[count - 1][1])]) > 0:
                sol[count - 1][1] = sol[count][1]
                sol[count][0] = 0
                flag = 1
            sol[count][1] = 0

    if sol[count][1] == 0:
        sol[count][0] = 0

    return sol

def get_coef_array_buy(peaks: np.array,
                       prices: np.array,
                       coefficient_power: int) -> np.array:
    perfect_buy = np.where(peaks == 0)[0]
    perfect_sell = np.where(peaks == 2)[0]
    prices = np.array(prices)
    buy_intervals = np.diff(np.concatenate([perfect_buy, [perfect_sell[-1] + 1]]))
    sell_intervals = np.diff(np.concatenate([[perfect_buy[0] - 1], perfect_sell]))
    perfect_sell_prices = np.repeat(prices[perfect_sell], buy_intervals)
    perfect_buy_prices = np.repeat(prices[perfect_buy[:len(perfect_sell)]], sell_intervals)
    coef_array_buy = (perfect_sell_prices - prices[perfect_buy[0]: len(perfect_sell_prices) + perfect_buy[0]]) / (
            perfect_sell_prices - perfect_buy_prices)
    coef_array_buy = np.power(coef_array_buy, coefficient_power)
    coef_array_buy = np.concatenate(
        [np.zeros(perfect_buy[0]), coef_array_buy, np.zeros(len(prices) - len(coef_array_buy) - perfect_buy[0])])
    return coef_array_buy


def get_coef_array_sell(peaks: np.array,
                        prices: np.array,
                        coefficient_power: int) -> np.array:
    perfect_buy = np.where(peaks == 0)[0]
    perfect_sell = np.where(peaks == 2)[0]
    prices = np.array(prices)
    buy_intervals = np.diff(np.concatenate([perfect_buy, [perfect_sell[-1] + 1]]))
    sell_intervals = np.diff(np.concatenate([[perfect_buy[0] - 1], perfect_sell]))
    perfect_sell_prices = np.repeat(prices[perfect_sell], buy_intervals)
    perfect_buy_prices = np.repeat(prices[perfect_buy[:len(perfect_sell)]], sell_intervals)
    coef_array_sell = (prices[perfect_buy[0]: len(perfect_sell_prices) + perfect_buy[0]] - perfect_buy_prices) / (
            perfect_sell_prices - perfect_buy_prices)
    coef_array_sell = np.power(coef_array_sell, coefficient_power)
    coef_array_sell = np.concatenate(
        [np.zeros(perfect_buy[0]), coef_array_sell, np.zeros(len(prices) - len(coef_array_sell) - perfect_buy[0])])
    return coef_array_sell


def get_buy_scale(coef_array_sum: np.array,
				  boundary: float,
				  option: int = 2) -> np.array:
	"""Option 1:
	sum(buy) = sum(nothing)*a = sum(sell)*b
	a = sum(buy)/sum(nothing)
	b = sum(buy)/sum(sell)
	return [..., buy, buy, nothing*a, nothing*a...,buy, sell*b...]

	Option 2:
	sum(buy, sell) = sum(nothing) * a
	"""
	if option == 1:
		sum_buy = np.sum(np.abs(coef_array_sum[np.where(coef_array_sum < -boundary)]))
		sum_nothing = np.sum(np.cos(1 / boundary * np.arccos(boundary) * np.abs(
			coef_array_sum[np.where((coef_array_sum >= -boundary) & (coef_array_sum <= boundary))])))
		sum_sell = np.sum(coef_array_sum[np.where(coef_array_sum > boundary)])
		a = sum_buy / sum_nothing
		b = sum_buy / sum_sell
		# Now we are changing values of our weights with respect to calculated a and b
		result = copy.deepcopy(coef_array_sum)
		result[np.where(coef_array_sum < -boundary)] = np.abs(
			coef_array_sum[np.where(coef_array_sum < -boundary)])  # buy
		result[np.where(coef_array_sum > boundary)] = coef_array_sum[np.where(coef_array_sum > boundary)] * b  # sell
		result[np.where((coef_array_sum >= -boundary) & (coef_array_sum <= boundary))] = np.cos(
			1 / boundary * np.arccos(boundary) * coef_array_sum[np.where(
				(coef_array_sum >= -boundary) & (coef_array_sum <= boundary))]) * a  # nothing
	elif option == 2:
		sum_action = np.sum(np.abs(coef_array_sum[np.where(
			(coef_array_sum < -boundary) | (coef_array_sum > boundary))]))  # weights sum for any action
		sum_nothing = np.sum(np.cos(1 / boundary * np.arccos(boundary) * np.abs(
			coef_array_sum[
				np.where((coef_array_sum >= -boundary) & (coef_array_sum <= boundary))])))  # weights sum for do nothing
		a = sum_action / sum_nothing
		# Now we are changing weights for do nothing
		result = copy.deepcopy(coef_array_sum)
		result[np.where((coef_array_sum < -boundary) | (coef_array_sum > boundary))] = np.abs(coef_array_sum[np.where(
			(coef_array_sum < -boundary) | (coef_array_sum > boundary))])  # buy and sell
		result[np.where((coef_array_sum >= -boundary) & (coef_array_sum <= boundary))] = np.cos(
			1 / boundary * np.arccos(boundary) * coef_array_sum[np.where(
				(coef_array_sum >= -boundary) & (coef_array_sum <= boundary))]) * a  # nothing
	return result / np.mean(result)  # weights