import time
import logging
import pickle as pkl

from pathlib import Path

from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config


def main() -> None:

	eval_results_dict = {}

	for period in ["train", "validation", "test"]: 

		logger.info(f"Evaluating {period} period.")

		# Load the config.
		run_dir_path = Path("/home/wuhlmann/BA/repo/lamah-ce_lstm/models/q_pred_landcover_0903_135428")
		cfg = Config(run_dir_path / "config.yml")

		# Set up tester and get the baseline NSE, with no attributes altered.
		tester = get_tester(cfg=cfg, run_dir=run_dir_path, period=period, init_model=True)
		
		logger.info(f"Tester initialized for {tester.period} period with basins {tester.basins}")
		logger.info("Conducting forward pass to calculate metrics.")
		raw_results = tester.evaluate(epoch=16, save_results=False, metrics=["NSE", "KGE", "MSE", "RMSE"])

		eval_results_dict[period] = raw_results

	domain_eval_results_path = Path(f"/home/wuhlmann/BA/data/processed_data/evaluation/{run_dir_path.stem}_domain_metrics.p")
	
	try: 
		with open(domain_eval_results_path, "wb") as out: 
			pkl.dump(eval_results_dict, out)	
		logger.info(f"Successfully domain evaluation results at {domain_eval_results_path}")
	
	except Exception as e: 

		logger.error(f"Pickling of domain evaluation results failed:\n{e}")

if __name__ == "__main__":

	logging.basicConfig(
		level=logging.INFO,              
		format="%(asctime)s - %(levelname)s - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)
	logger = logging.getLogger(__name__)
	logging.getLogger("neuralhydrology").setLevel(logging.CRITICAL)

	start_time = time.time()

	main()

	elapsed = time.time() - start_time
	hours = int(elapsed // 3600)
	minutes = int((elapsed % 3600) // 60)
	seconds = elapsed % 60

	logger.info(f"Domain evaluation completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")