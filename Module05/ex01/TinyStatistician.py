class TinyStatistician:
    def mean(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        return float(sum(x) / len(x))
    
    def median(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        for v in x:
            if not isinstance(v, (int, float)):
                return None
        x_sorted = sorted(x)
        mid = len(x) // 2
        if len(x) % 2 == 1: # impair
            return float(x_sorted[mid])
        else: # pair
            return float((x_sorted[mid - 1] + x_sorted[mid]) / 2)

    def quartile(self, x):
        q1 = self.percentile(x, 25)
        q3 = self.percentile(x, 75)
        if q1 is None or q3 is None:
            return None
        return [float(q1), float(q3)]

    def percentile(self, x, p):
        if not isinstance(x, list) or len(x) == 0 or p < 0 or p > 100 or not isinstance(p, (int, float)):
            return None
        for v in x:
            if not isinstance(v, (int, float)):
                return None
        x_sorted = sorted(x)
        k = (len(x_sorted) - 1) * (p / 100) # idx du percentile
        f = int(k)
        d = k - f
        if f + 1 < len(x_sorted):
            return float(x_sorted[f] + d * (x_sorted[f + 1] - x_sorted[f]))
        else:
            return float(x_sorted[f])

    def var(self, x):
        if not isinstance(x, list) or len(x) == 0:
            return None
        m = self.mean(x)
        total = sum((v - m)**2 for v in x)
        return float(total / (len(x) - 1))

    def std(self, x):
        variance = self.var(x)
        if variance is None:
            return None
        return float(variance ** 0.5)

def main():
    """Tester of my TinyStatistician class"""
    print("============= TEST  ===================")
    a = [1, 42, 300, 10, 59]
    print(a)
    print("mean : ", TinyStatistician().mean(a))
    print("expected : 82.4")
    
    print("median : ", TinyStatistician().median(a))
    print("expected : 42.0")

    print("quartile : ", TinyStatistician().quartile(a))
    print("expected : [10.0, 59.0]")

    print("percentile 10 : ", TinyStatistician().percentile(a, 10))
    print("expected : 4.6")

    print("percentile 15 : ", TinyStatistician().percentile(a, 15))
    print("expected : 6.4")

    print("percentile 20 : ", TinyStatistician().percentile(a, 20))
    print("expected : 8.2")

    print("variance : ", TinyStatistician().var(a))
    print("expected : 12279.439999999999")

    print("ecart type : ", TinyStatistician().std(a))
    print("expected : 110.81263465868862")

    return 0

if __name__ == "__main__":
    main()