import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(__file__).resolve().parents[1]
if len(sys.argv)!=3:raise SystemExit('usage: parse_hls_report.py fp32_csynth.xml int8_csynth.xml')
def parse(path):
 t=ET.parse(path).getroot()
 def val(*paths):
  for p in paths:
   node=t.find(p)
   if node is not None and node.text is not None:return node.text.strip()
  return None
 return {'report':str(Path(path).resolve()),'latency_cycles_min':val('.//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency'),'latency_cycles_max':val('.//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency'),'interval_cycles_min':val('.//PerformanceEstimates/SummaryOfOverallLatency/Interval-min'),'interval_cycles_max':val('.//PerformanceEstimates/SummaryOfOverallLatency/Interval-max'),'estimated_clock_ns':val('.//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'),'DSP':val('.//AreaEstimates/Resources/DSP'),'LUT':val('.//AreaEstimates/Resources/LUT'),'FF':val('.//AreaEstimates/Resources/FF'),'BRAM_18K':val('.//AreaEstimates/Resources/BRAM_18K')}
res={'status':'measured HLS synthesis','fp32':parse(sys.argv[1]),'int8':parse(sys.argv[2])}
(root/'results/synthesis_results.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
