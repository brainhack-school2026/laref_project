import sys
import classifier
import train
import analysis
import data_handler
import data_import


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [fetch|process|train]")
        sys.exit(1)

    command = sys.argv[1]  

    match command:
        case "fetch":
            data_import.get_all_data()

        case "process":
            subjects = data_handler.load_all_subjects()
            harmonized = data_handler.harmonize(subjects)
            data_handler.prepare_data(harmonized)

        case "train":
            train.train()

        case "analyse":
            analysis.analyse()


        case _:
            print(f"Unknown command: '{command}'. choose between : fetch, process, train, analyse")
            sys.exit(1)


if __name__ == "__main__":
    main()