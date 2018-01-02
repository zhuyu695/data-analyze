from HistoricalAnalyze import HistoricalAnalyze
import matplotlib.pyplot as plt
from CurrentAnalyze import CurrentAnalyze
from Util import Util

# historicalAnalyze = HistoricalAnalyze()
# historicalAnalyze.run_example()

# plt.interactive(False)
currentAnalyze = CurrentAnalyze()
# print(currentAnalyze.get_latest_price_df(CurrentAnalyze.symbols, 'ethereum'))
# print(currentAnalyze.get_candle_stick('XRP', 'ethereum', '3m'))
# Util.df_scatters(currentAnalyze.get_candle_sticks(['XRP','XVG','IOTA','WTC','SUB','ICX','NEO', 'WAVES'], 'ethereum', '1h'), 'cumulative prodct')
currentAnalyze.chart_all()
# currentAnalyze.get_depth_chart('POEETH', num=1000, action='sell')