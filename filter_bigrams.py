import io

N = 5000
bigrams = []

with io.open("w2_.txt", encoding="latin1") as f:
    for line in f:
        elements = [e.strip() for e in line.split("\t")]
        count = int(elements[0])
        bigram = elements[1:3]
        bigrams.append((bigram, count))

sorted_bigrams = sorted(bigrams, key=lambda b: -b[1])
sorted_bigrams = sorted_bigrams[:N]

with io.open("w2_%s.txt" % N, mode="w", encoding="utf8") as f:
    for bigram, count in sorted_bigrams:
        f.write("%s\t%s\t%s\n" % (count, bigram[0], bigram[1]))
