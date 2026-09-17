//+------------------------------------------------------------------+
//|                                        AdaptiveVolMomentum.mq5   |
//| ATR-normalized momentum, Z-scored, optionally blended with an   |
//| RSI-based range component via the Kaufman Efficiency Ratio.     |
//| ATR/RSI use Wilder RMA smoothing so results match the MQL4 and  |
//| TradingView (Pine) ports of this same indicator bar-for-bar.    |
//+------------------------------------------------------------------+
#property copyright ""
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   2
#property indicator_label1  "AdaptiveVolMomentum"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2
#property indicator_label2  "Signal"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrSilver
#property indicator_width2  1
#property indicator_level1  2.0
#property indicator_level2  -2.0
#property indicator_level3  0.0
#property indicator_levelcolor clrGray
#property indicator_levelstyle STYLE_DOT
#property indicator_minimum -4.5
#property indicator_maximum  4.5

input int    InpMomentumPeriod = 14;    // モメンタム計算期間
input int    InpATRPeriod      = 14;    // ATR期間（Wilder RMA）
input int    InpZPeriod        = 100;   // Zスコア算出用のローリング窓
input bool   InpUseERBlend     = true;  // Kaufman効率比によるハイブリッド化
input int    InpERPeriod       = 14;    // 効率比(ER)期間
input int    InpRSIPeriod      = 14;    // レンジ成分に使うRSI期間
input int    InpSmoothPeriod   = 3;     // 最終値の平滑化EMA期間
input double InpClip           = 4.0;   // 出力のクリップ範囲(±)

double BufMain[];
double BufSignal[];
double BufATR[];
double BufAvgGain[];
double BufAvgLoss[];

int WarmupBars()
{
   int m = InpMomentumPeriod;
   if(InpATRPeriod > m) m = InpATRPeriod;
   if(InpERPeriod  > m) m = InpERPeriod;
   if(InpRSIPeriod > m) m = InpRSIPeriod;
   return m + InpZPeriod;
}

double RangeComponent(const double rsi)
{
   double v = (rsi - 50.0) / 12.5;
   if(v > InpClip)  v = InpClip;
   if(v < -InpClip) v = -InpClip;
   return v;
}

int OnInit()
{
   SetIndexBuffer(0, BufMain,    INDICATOR_DATA);
   SetIndexBuffer(1, BufSignal,  INDICATOR_DATA);
   SetIndexBuffer(2, BufATR,     INDICATOR_CALCULATIONS);
   SetIndexBuffer(3, BufAvgGain, INDICATOR_CALCULATIONS);
   SetIndexBuffer(4, BufAvgLoss, INDICATOR_CALCULATIONS);

   int warmup = WarmupBars();
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, warmup);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, warmup);
   IndicatorSetString(INDICATOR_SHORTNAME, "AdaptiveVolMomentum");
   IndicatorSetInteger(INDICATOR_DIGITS, 3);
   return INIT_SUCCEEDED;
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   int warmup = WarmupBars();
   if(rates_total < warmup + 2)
      return 0;

   int start = (prev_calculated > 1) ? prev_calculated - 1 : 1;

   for(int i = start; i < rates_total; i++)
   {
      // True Range -> Wilder ATR (recursive, must persist across calls)
      double tr = MathMax(high[i]-low[i], MathMax(MathAbs(high[i]-close[i-1]), MathAbs(low[i]-close[i-1])));
      if(i == 1)
         BufATR[i] = tr;
      else
         BufATR[i] = BufATR[i-1] + (tr - BufATR[i-1]) / InpATRPeriod;

      // Wilder RSI building blocks (recursive)
      double gain = MathMax(close[i]-close[i-1], 0.0);
      double loss = MathMax(close[i-1]-close[i], 0.0);
      if(i == 1)
      {
         BufAvgGain[i] = gain;
         BufAvgLoss[i] = loss;
      }
      else
      {
         BufAvgGain[i] = BufAvgGain[i-1] + (gain - BufAvgGain[i-1]) / InpRSIPeriod;
         BufAvgLoss[i] = BufAvgLoss[i-1] + (loss - BufAvgLoss[i-1]) / InpRSIPeriod;
      }

      if(i < warmup)
      {
         BufMain[i]   = 0.0;
         BufSignal[i] = 0.0;
         continue;
      }

      // --- design①: ATR-normalized momentum, Z-scored over InpZPeriod ---
      double sum = 0.0, sumSq = 0.0, normMomAtI = 0.0;
      for(int k = i - InpZPeriod + 1; k <= i; k++)
      {
         double nm = (BufATR[k] > 0.0) ? (close[k] - close[k-InpMomentumPeriod]) / BufATR[k] : 0.0;
         sum   += nm;
         sumSq += nm*nm;
         if(k == i) normMomAtI = nm;
      }
      double mean     = sum / InpZPeriod;
      double variance = sumSq/InpZPeriod - mean*mean;
      if(variance < 0.0) variance = 0.0;
      double stdev  = MathSqrt(variance);
      double trendZ = (stdev > 0.0) ? (normMomAtI - mean) / stdev : 0.0;
      if(trendZ > InpClip)  trendZ = InpClip;
      if(trendZ < -InpClip) trendZ = -InpClip;

      double final_;
      if(InpUseERBlend)
      {
         // --- range component: RSI recentred to the same +/-Clip scale ---
         double avgLoss = BufAvgLoss[i];
         double rsi;
         if(avgLoss == 0.0)
            rsi = (BufAvgGain[i] == 0.0) ? 50.0 : 100.0;
         else
         {
            double rs = BufAvgGain[i] / avgLoss;
            rsi = 100.0 - 100.0 / (1.0 + rs);
         }
         double rangeComp = RangeComponent(rsi);

         // --- design②: Kaufman efficiency ratio, 1 = pure trend, 0 = pure noise ---
         double direction  = MathAbs(close[i] - close[i-InpERPeriod]);
         double volatility = 0.0;
         for(int k = i - InpERPeriod + 1; k <= i; k++)
            volatility += MathAbs(close[k] - close[k-1]);
         double er = (volatility > 0.0) ? direction / volatility : 0.0;

         final_ = er*trendZ + (1.0-er)*rangeComp;
      }
      else
         final_ = trendZ;

      BufMain[i] = final_;

      double alpha = 2.0 / (InpSmoothPeriod + 1.0);
      if(i == warmup)
         BufSignal[i] = final_;
      else
         BufSignal[i] = BufSignal[i-1] + alpha*(final_ - BufSignal[i-1]);
   }
   return rates_total;
}
