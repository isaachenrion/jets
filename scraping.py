from loggers import StatsLogger
import os
import csv
def scrape_results(model_dir):
    model_filenames = [os.path.join(model_dir, fn) for fn in os.listdir(model_dir)]
    for i, fn in enumerate(model_filenames):
        with open(os.path.join(fn, 'stats.csv'), newline='') as f:
            reader = csv.DictReader(f)
            lines = [l for l in reader]
            if i == 0:
                headers = lines[0]
                sl = StatsLogger(os.path.join(model_dir,'summary'), headers)
            sl.log(lines[-1])

def main():
    base = 'finished_models'
    #models = ['mpnn', 'recnn', 'relation']
    flavours = ['vanilla', 'set', 'id']
    iters = [1, 2, 3]

    model_dirs = [os.path.join(base, 'mpnn', flavour, str(i)) for flavour in flavours for i in iters]

    model_dirs.extend([os.path.join(base, 'recnn/simple'), os.path.join(base, 'relation')])

    for md in model_dirs: print(md)
    for model_dir in model_dirs:
        scrape_results(model_dir)

if __name__ == '__main__':
    main()
