import sys
import csv

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: {} file1.csv file2.csv ...\n".format(sys.argv[0]))
        sys.exit(1)

    files = sys.argv[1:]
    data = {}
    all_x = set()

    for fn in files:
        mapping = {}
        with open(fn, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    x = int(row[0])
                    y = int(row[1])
                except ValueError:
                    continue
                mapping[x] = y
                all_x.add(x)
        data[fn] = mapping

    sorted_x = sorted(all_x)
    header = ["#"] + [f'{str(i):20s}' for i in files]
    print("\t".join(header))
    for x in sorted_x:
        row = [str(x)]
        for fn in files:
            val = data[fn].get(x, "")
            row.append(f'{str(val):20s}')
        print("\t".join(row))


if __name__ == "__main__":
    main()
