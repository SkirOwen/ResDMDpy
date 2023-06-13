import rdp.config
from rdp import logger


def main():
	args = rdp.config.parse_args()

	if args.log_level:
		logger.setLevel(args.log_level)

	logger.info(f"Plotting: {args.plot_graph}")

	if args.example:
		if args.example.lower() == "cylinder":
			from rdp.examples.cylinder_wake import run
			run(args.modes, args.plot_graph)


if __name__ == "__main__":
	main()
