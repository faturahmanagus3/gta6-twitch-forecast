# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/faturahmanagus3/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/faturahmanagus3/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecaster for GTA VI's launch-day Twitch viewership. Given a case whose only known figure is hour 1's viewer count, it forecasts hours 2-24 of the release day as five quantiles (5%, 25%, 50%, 75%, 95%) per hour. The forecast is the weighted empirical quantile curve of past big launches' release days, carried inside each case as comparables: for every comparable launch it takes the log-ratio of each hour to that launch's own hour 1, weights launches by how similar their hour-1 scale is to this case's, smooths each quantile level over neighbouring hours, and scales the result by this case's hour-1 print. Standard library only, deterministic, offline. On leave-one-out over the 37 comparable launches with complete release-day curves it scores a mean log-scale pinball loss of 0.413 against the persistence baseline's 0.930, winning 32 of 37.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dontol&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
