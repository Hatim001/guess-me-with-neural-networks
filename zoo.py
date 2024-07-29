import csv


class Zoo:
    def __init__(self, src):
        self.questions = []
        self.targets = []
        self.qas = []

        self.src = src

        with open(self.src, "r") as zoocsv:
            entries = list(csv.reader(zoocsv))
            labels = entries[0]
            self.questions = labels[1:]
            duplicates = 0
            for row in entries[1:]:
                tmp_qas = list(map(int, row[1:]))
                if tmp_qas not in self.qas:  # remove indistinguishable entries
                    self.targets.append(row[0])
                    self.qas.append(list(map(int, row[1:])))
                else:
                    duplicates += 1

            print(f"\nloading {src}...")
            print(f"{len(self.targets)} targets ({duplicates} duplicates removed)")
            print(f"{len(self.questions)} questions")

    def get_answer(self, _question, _target):
        for target in self.targets:
            if target == _target:
                for question in self.questions:
                    if question == _question:
                        return self.qas[self.targets.index(target)][
                            self.questions.index(question)
                        ]
                break
        return None

    def get_answer_i(self, qi, ti):
        ans = self.qas[ti][qi]
        return 1 if ans == 1 else -1

    def animal_in_list(self, _target):
        return _target in self.targets

    def test(self):
        target = input("Enter animal: ")
        while not self.animal_in_list(target):
            print("Animal not in list. Please pick another animal...")
            target = input("Enter animal: ")

        while True:
            question = input("Question: ")
            answer = self.get_answer(question, target)
            if answer == 1:
                print("Yes")
            elif answer == 0:
                print("No")
            else:
                print("Unknown question")
