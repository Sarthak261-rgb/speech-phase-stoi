"""Run the whole study end to end:  python run_all.py
Creates outputs/{audio,figures,tables,challenge} and outputs/results.json (roughly half an hour)."""
import time
import experiments as E

if __name__ == "__main__":
    t0 = time.time()
    E.run_all()
    print(f"finished in {(time.time() - t0) / 60:.1f} min")
