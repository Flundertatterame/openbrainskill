import csv


def majority_vote(labels):

    count = {}

    for x in labels:
        count[x] = count.get(x, 0) + 1

    max_count = max(count.values())

    candidates = [
        k for k,v in count.items()
        if v == max_count
    ]

    # tie:
    return sorted(candidates)[0]


def align_predictions(
    predictions,
    epoch_size=6
):

    aligned=[]

    for i in range(0,len(predictions),epoch_size):

        epoch = predictions[i:i+epoch_size]

        if len(epoch)==epoch_size:

            stage=majority_vote(epoch)

            aligned.append(stage)

    return aligned



def main():

    pred=[
        "2",
        "2",
        "1",
        "2",
        "2",
        "1"
    ]


    result=align_predictions(pred)

    print(result)


    with open(
        "epoch_alignment_test.csv",
        "w",
        newline=""
    ) as f:

        writer=csv.writer(f)

        writer.writerow(
            [
                "5s_predictions",
                "aligned_stage"
            ]
        )

        writer.writerow(
            [
                ",".join(pred),
                result[0]
            ]
        )


if __name__=="__main__":
    main()