import sys, numpy as np

def stats(path):
    a = np.load(path)
    return dict(path=path, n=a.size, min=float(np.min(a)), max=float(np.max(a)), mean=float(np.mean(a)), med=float(np.median(a)), p95=float(np.percentile(a,95)), p5=float(np.percentile(a,5)), amp=float(np.max(a)-np.min(a)))

if __name__ == '__main__':
    for p in sys.argv[1:]:
        try:
            s = stats(p)
            print(s)
        except Exception as e:
            print({'path': p, 'error': str(e)})
