from api_rate_limiter_rl.experiments import run_experiments


if __name__ == "__main__":
    result = run_experiments()
    print(result["report_path"])

