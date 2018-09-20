#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Helpers Methods for Trading Strategy Development Workflow
# http://github.com/ranaroussi/futuresio-webinars
#
# Copyright 2017 Ran Aroussi
#
# Licensed under the GNU Lesser General Public License, v3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/lgpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.dates as mdates
try:
    from mpl_finance import candlestick_ohlc
except ImportError:
    from matplotlib.finance import candlestick_ohlc


def compsum(df):
    return (1 + df).cumprod() - 1


def train_test_split(df, test_size=.2):
    train_size = int(len(df) * (1 - test_size))
    return df[:train_size], df[train_size:]


def subplot(dfs, title="", upper_plot_ratio=3, figsize=None, *args, **kwargs):
    colors = ['#af4b64', '#3399cc', '#4fa487',
              '#9b59b6', '#95a5a6', '#ee8659', '#4f6c86']

    height_ratios = [upper_plot_ratio] + [1] * (len(dfs) - 1)

    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(dfs), 1, height_ratios=height_ratios)

    while len(dfs) > len(colors):
        colors = colors * 2

    i = 0
    axs = {}
    for name, df in dfs.items():
        if i == 0:
            axs[i] = plt.subplot(gs[i], **kwargs)
            axs[i].set_title(title, fontsize=14, fontweight='bold')
        else:
            axs[i] = plt.subplot(gs[i], sharex=axs[0])

        axs[i].plot(df, color=colors[i], **kwargs)
        axs[i].set_ylabel(name)

        i += 1

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.close()
    return fig


def plot_candlestick(df, ax=None, fmt="%Y-%m-%d", cols=("open", "high", "low", "close")):
    if ax is None:
        fig, ax = plt.subplots()

    idx_name = df.index.name
    dat = df.reset_index()[[idx_name] + cols]
    dat[df.index.name] = dat[df.index.name].map(mdates.date2num)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.xticks(rotation=45)
    _ = candlestick_ohlc(ax, dat.values, width=.6, colorup='g', alpha=1)
    ax.set_xlabel(idx_name)
    ax.set_ylabel("Price")
    return ax


def sharpe(returns, riskfree=0, periods=252):
    returns = returns[returns != 0].dropna()
    try:
        return np.sqrt(periods) * (np.mean(returns - riskfree)) / np.std(returns)
    except:
        return 0.

def cagr(returns):
    years = ((returns.index.max() - returns.index.min()).days)/365
    roi = 1 + compsum(returns).values[-1]
    return roi ** (1 / years) - 1.

def win_rate(returns):
    returns.fillna(0, inplace=True)
    total_trades = len(returns[returns != 0])
    wins = len(returns[returns > 0])

    if total_trades == 0:
        return 0

    return (wins / total_trades)


def avg_return(returns):
    returns.fillna(0, inplace=True)
    return returns[returns != 0].mean()

def avg_win(returns):
    returns.fillna(0, inplace=True)
    return returns[returns > 0].mean()

def avg_loss(returns):
    returns.fillna(0, inplace=True)
    return returns[returns < 0].mean()


def profit_factor(returns):
    returns.fillna(0, inplace=True)
    wins = sum(returns[returns > 0])
    loses = sum(returns[returns < 0])

    if loses == 0:
        return 0.
    return ( wins / abs(loses) )


def drawdown(returns, daily=False):
    if daily:
        returns = returns.resample("D").sum()

    cumret = returns.fillna(0)

    dd = pd.DataFrame(index=cumret.index, data={"percent": cumret.fillna(0)})
    dd['duration'] = np.where(dd['percent'] < 0, 1, 0)
    dd['duration'] = dd['duration'].groupby(
        (dd['duration'] == 0).cumsum()).cumcount()

    return dd


def recovery_factor(returns, dd):
    max_dd = abs(dd.min())
    total_returns = sum(returns)
    return total_returns/max_dd if max_dd != 0 else np.nan


def colormap2d(x, y, res,
               x_title=None, y_title=None,
               title="Color Map",
               figsize=(8, 6)):

    if not x_title:
        x_title = "X"
    if not y_title:
        y_title = "Y"

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap('jet')
    plt.pcolormesh(y, x, res, cmap=cmap)
    plt.title(title, fontweight="bold")
    plt.xlabel(y_title)
    plt.ylabel(x_title)
    plt.colorbar()
    plt.show()


def unravel2d(x, y, res):
    xi, yi = np.unravel_index(res.argmax(), res.shape)
    return x[xi], y[yi]


def colormap3d(x, y, z,
               x_title=None, y_title=None, z_title=None,
               title="Color Map",
               figsize=(12, 10)):

    if not x_title:
        x_title = "X"
    if not y_title:
        y_title = "Y"
    if not z_title:
        z_title = "Z"

    cmap = plt.cm.get_cmap('jet')
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel("\n\n" + x_title)
    ax.set_ylabel("\n\n" + y_title)
    ax.set_zlabel(" " + z_title)
    plt.title(title + "\n", fontweight="bold")
    plt.show()


def iterate(df, skip=1, progress=True):
    """
    example:
    for idx, loc in atw.iterate(df):
        print(line)
    """
    if progress:
        pbar = ProgressBar(len(df))

    for idx, _ in df.iterrows():
        loc = df.index.get_loc(idx)
        if loc >= skip:
            yield idx, loc
        if progress:
            pbar.animate()
            time.sleep(0.000001)

    print('')
    return


def sum_by_hour(data):
    hours = pd.DataFrame(data.resample('H').sum()).fillna(0)
    hours['hour'] = hours.index
    hours['hour'] = hours['hour'].apply(lambda d: d.hour)
    return hours.groupby('hour').sum().fillna(0)


def sum_by_day(data):

    weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    days = pd.DataFrame(data.resample('D').sum()).fillna(0)
    days['day'] = days.index
    days['day'] = days['day'].apply(lambda d: d.day)

    for ix in range(0, 7):
        days.loc[days.day == ix, 'weekday'] = weekdays[ix]

    days = days.groupby('weekday').sum()

    # sort
    days['day'] = days.index
    for ix, day in enumerate(weekdays):
        days.loc[days.index == day, 'day'] = ix
    days = days.sort_values('day')[['return']]

    return days


def sum_by_week(data):
    weeks = pd.DataFrame(data.resample('W').sum()).fillna(0)
    weeks.index.rename('weeks', inplace=True)
    weeks['week'] = weeks.index
    weeks['week'] = weeks['week'].apply(lambda d: d.week)
    return weeks.groupby('week').sum().fillna(0)


def sum_by_month(data, total=True):
    months = pd.DataFrame(data.resample('M').sum()).fillna(0)
    months['month'] = months.index
    months['month'] = months['month'].apply(lambda d: d.month)
    months = months.groupby('month').sum().T
    months.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.T.fillna(0)


def sum_by_quarter(data):
    quarters = pd.DataFrame(data.resample('W').sum()).fillna(0)
    quarters.index.rename('quarters', inplace=True)
    quarters['quarter'] = quarters.index
    quarters['quarter'] = quarters['quarter'].apply(lambda d: d.quarter)
    return quarters.groupby('quarter').sum().fillna(0)


def sum_by_year(data):
    """ returns a dataframe of year returns of passed data """
    years = pd.DataFrame(data.resample('A').sum())
    years['year'] = years.index
    years['year'] = years['year'].apply(lambda d: d.year)
    years.set_index('year', inplace=True)
    return years.fillna(0)


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)
        self.elapsed = 1

    def animate(self, iteration=None):
        if iteration is None:
            self.elapsed += 1
            iteration = self.elapsed
        else:
            self.elapsed += iteration

        print('\r'+str(self), end='')
        sys.stdout.flush()
        self.update_iteration()

    def update_iteration(self):
        self.__update_amount((self.elapsed / float(self.iterations)) * 100.0)
        self.prog_bar += '  %s of %s complete' % (
            self.elapsed, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * \
            num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


def montecarlo(series, sims=100, bust=-1, goal=0):

    if not isinstance(series, pd.Series):
        raise ValueError("Data must be a Pandas Series")

    class __make_object__:
        """Monte Carlo simulation results"""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def plot(title="Monte Carlo Simulation Results", figsize=None):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cumsum, lw=1, alpha=.8)
        ax.plot(cumsum["original"], lw=3, color="r",
                alpha=.8, label="Original")
        ax.axhline(0, color="black")
        ax.legend()
        ax.set_title(title, fontweight="bold")
        plt.ylabel("Results")
        plt.xlabel("Occurrences")
        plt.show()
        plt.close()

    results = [series.values]
    for i in range(1, sims):
        results.append(series.sample(frac=1).values)

    df = pd.DataFrame(results).T
    df.rename(columns={0: 'original'}, inplace=True)

    cumsum = df.cumsum()
    total = cumsum[-1:].T
    dd = cumsum.min()[cumsum.min() < 0]
    nobust = cumsum[cumsum.min()[cumsum.min() > -abs(bust)].index][-1:]

    return __make_object__(**{
        "data": df,
        "stats": {
            "min": total.min().values[0],
            "max": total.max().values[0],
            "mean": total.mean().values[0],
            "median": total.median().values[0],
            "std": total.std().values[0],
            "maxdd": dd.min(),
            "bust": len(dd[dd <= -abs(bust)]) / sims,
            "goal": (nobust >= abs(goal)).sum().sum() / sims,
        },
        "maxdd": {
            "min": dd.min(),
            "max": dd.max(),
            "mean": dd.mean(),
            "median": dd.median(),
            "std": dd.std()
        },
        "plot": plot
    })


def monthly_returns(returns,
                            title="Monthly Returns (%)",
                            title_color="black",
                            title_size=12,
                            annot_size=10,
                            figsize=None,
                            cmap='RdYlGn',
                            cbar=True,
                            square=False):

    # resample to business month
    returns = returns.resample('BMS').sum()

    # get close / first column if given DataFrame
    if isinstance(returns, pd.DataFrame):
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and 'close' in returns.columns:
            returns = returns['close']
        else:
            returns = returns[returns.columns[0]]

    # get returnsframe
    returns = pd.DataFrame(data={'Returns': returns})
    returns['Year'] = returns.index.strftime('%Y')
    returns['Month'] = returns.index.strftime('%b')

    # make pivot table
    returns = returns.pivot('Year', 'Month', 'Returns').fillna(0)

    # add missing months
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # order columns by month
    returns = returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

    returns *= 100

    return returns


def monthly_returns_heatmap(returns,
                            title="Monthly Returns (%)",
                            title_color="black",
                            title_size=12,
                            annot_size=10,
                            figsize=None,
                            cmap='RdYlGn',
                            cbar=True,
                            square=False):

    returns = monthly_returns(returns, title, title_color, title_size,
                            annot_size, figsize, cmap, cbar, square)
    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[0] // 2)
        plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(returns, ax=ax, annot=True,
                     annot_kws={"size": annot_size}, fmt="0.2f", linewidths=0.5,
                     square=square, cbar=cbar, cmap=cmap)
    ax.set_title(title, fontsize=title_size,
                 color=title_color, fontweight="bold")

    fig.subplots_adjust(hspace=0)
    plt.yticks(rotation=0)
    plt.show()
    plt.close()


class _timer():
    def __init__(self):
        self._TIMER_START_ = 0
        self._TIMER_END_ = 0

    def start(self):
        self._TIMER_START_ = datetime.datetime.now()
        self._TIMER_END_ = self._TIMER_START_

    def reset(self):
        return self._timeit(reset=False)

    def stop(self):
        return self._timeit(reset=True)

    def _timeit(self, reset=True):
        if self._TIMER_START_ == 0:
            return 'Error: Timer need to be initilied using timer.start()'

        self._TIMER_END_ = datetime.datetime.now()
        time_delta = self._TIMER_END_ - self._TIMER_START_

        if (reset == True):
            self._TIMER_START_ = datetime.datetime.now()
            self._TIMER_END_ = self._TIMER_START_

        seconds = time_delta.total_seconds()
        milliseconds = seconds * 1000
        microseconds = milliseconds * 1000
        # nanoseconds = microseconds * 1000

        if seconds >= 1:
            print('\n[*] Total runtime: %.2f s' % seconds)
        # elif nanoseconds >= 1000:
        #     print('\n[*] Total runtime: %.2f ns' % seconds)
        elif microseconds >= 1000:
            print('\n[*] Total runtime: %.2f ms' % milliseconds)
        else:
            print('\n[*] Total runtime: %.2f \xB5s' % microseconds)

        return


timer = _timer()
