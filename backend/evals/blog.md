The following is a sloppy first pass.

# Building Evals for Image Generation
Over the last few years, the Data and ML field has seen an influx of people from traditional software engineering backgrounds. As they go to build products backed by generative models, they're faced with a daunting word:

Evals.

As one of those people, I can attest. The goal of this blog is to demystify evals, and show the real steps that one could go through to build evals for an image generation model.

### How to think about evals

Evals are how data teams quantitatively measure the performance of their models. 

Evals are to data teams as testing in CI is to software engineers. The critical difference is that unlike test, evals are not strictly pass/fail; each one is meant to quantify the model's performance on a certain dimension.

Think of each eval as a stat on a character stat sheet.

[insert character customizeation screen with sliders]

Taking LLM evals as an example, some standard evals are:

HumanEval: Tests the "coding ability" stat
GSM8K: Tests the "mathematical reasoning" stat
MMLU: Tests the "knowledge (science, history, law, medicine, ethics, etc)" stat 
HellaSwag: Tests the "social intelligence" stat

Evals are used for experimentation; the continuous challenge to push the numbers up.

Evals can also be used for quality control; similar to software tests, ensuring that your new experiments wouldn't regress production if merged.

When you actually look at their code, you'll see that they can be anything that measures correctness or quality! This is the thing to demystify. Evals are just code, code that you can write, no fancy witchcraft, to assess the correctness of an output. You can even use other models to eval your models, something you'll see multiple times in this blog.

### Our project
We've built an aesthetic QR code generator. 

[Qart](link)

It's an image diffusion service that uses a [controlnet](link) model alongside the diffusion model to take a functioning QR code and guide the image generation toward that code.

### What is the outcome?
In its most basic form, running an eval against our model will produce a score. 

That score lies on some spectrum such as a 0-100% range. If an eval scores 50%, that means the model's output passed 50% of the tests. If a model scores 100%, that means it's time to find a harder eval!

A term you'll come across often is the `pass@k` metric. This is the percentage of tests the model passed when given `k` samples generated for each test.

So `pass@1=50` means that given a set of tests, the model was ran once per test and for 50% of the tests it passed.
`pass@3=99` means that given a set of tests, the model was ran three times per test, and for 99% of the tests, at least one of those three attempts pass.

Ideally, each eval is narrowly scoped to test a certain dimension of performance.

In our case, we have two stats to measure:
1. QR Code scannability and accuracy: does the generated image actually scan? 
2. Aesthetics: is the generated image aesthetic or not?

And it's all down to our own project goals to decide how to balance these. Because while not mutually exclusive, specializing a model in one metric often makes it worse at others. 

Do we max out the QR code scannabilty stat, at the risk of making the images bland?
Or do we max out the aesthetics, and risk having mostly unscannable codes? 
Or how can we get the best of both?

This is a question question you'll see us explore as we implement our evals.

## Building our Evals

To have good evals, we need:
1. a way to measure the output of our model relative to what is "correct"
2. a way to record and reproduce these scores over time, so we can bravely experiment with new models / codebases and revert.

Let's start with measurement. We'll be building two evals:
1. QR Code scannability
2. Aesthetics
   
For both, the test harness will be simple:
```python
# generate many outputs
images = model.batch_generate()

# test each output
for image in images:
    passed: bool = do_test(image)

# calculate pass@ metrics
```

And our job is to get creative and figure out how to implement that do_test() function. 

How do you automate qr code scannability into a function? How do you automate aesthetic checking into a function? This is the crux of our project.


Despite scannability and aesthetics being wildly different things to measure, the workflow we go through to develop that do_test() function is actually quite similar:

1. build a dataset of model outputs
2. manually classify them into pass/fail buckets. This establishes "ground truth".
3. experiment with different do_test() implementations until we find the one that when given the dataset, classifies them the similarly to how we did.

If done well, we end up with a do_test() implementation that closely emulates our own sense of ground truth.

## Scannability
Our "Ground Truth" for scannability is whether or not an iphone can scan it.

The ultimate eval would be to set up a physical screen with an iphone pointed at it, and do_test(image) passes when the phone scans on the image.

We don't want to deal with the hardware and uptime of that setup, so instead we'll need to build a closest possible emulation of iphone scanning using pure software.

### Building a dataset
I ran a script that generates 100+ images on random prompts and QR code URLs `modal run backend.evals.scannability::generate_detector_set`.

It dumped them into `backend/evals/data/scannability`

### Sorting

I then sat there, for an hour, holding my iphone camera up to the screen to see if it scans. I sorted them into `valid` and `invalid` buckets based on if they scanned the correct QR.

From the set, the iphone was able to scan 85.3% of the images. 

Note that in an ideal world, we'd aim for a 50/50 split of valid to invalid.

### Experimenting with QR scanners

### OpenCV
`detect_qr_opencv` uses the qr detection library built into OpenCV.

I took the manually labeled dataset, and passed the images through OpenCV to see if it detected what the iphone did. This is when I first realized that emulating iphone scan ability will be harder than expected. 

OpenCV yielded:
```
Detector:        OpenCV
True Positives:  21
False Positives: 0
True Negatives:  19
False Negatives: 89
Score:           0.310
```

Where :
- "True Positive" = the detector correctly scanned a code that the iphone scanned
- "False Positive" = the detector scanned a code that iphone couldn't.
- "True Negative" = the detector couldn't scan a code, nor could the iphone
- "False Negative" = the detector couldn't scan a code that the iphone could.
- "Score" = Accuracy, proportion of True detections: (TP + TN) / (TP + TN + FP + FN)

Note that none of this has anything to do with if the QR itself is truly valid or invalid. Everything we're measuring here is relative to the ability of an iphone to scan it, since that's how users will interract with our product. 

It's great to see no false positives, because many false positives makes us incorrectly believe our model is generating iphone scannable QRs when it's not.

But there are quite a few false negatives, because that makes us incorrectly believe that our model is underperforming, which would likely result in a future where we steering the model toward more aesthetically boring generations for the sake of QR scannability.

Let's experiment with other scanning methods.

### Pyzbar
`detect_qr_pyzbar`
Pyzbar is a library built on top of openCV.
```
Detector:        Pyzbar
True Positives:  30
False Positives: 1
True Negatives:  18
False Negatives: 80
Score:           0.372
```
It performs equally dissapointingly.


### QReader
`class QReader`
Qreader is the first QR detector implementation that uses neural networks under the hood, specifically a YoloV8 object detection model. 

Yes, it's ok to use models to eval models! If the goal is to emulate ground truth, often times using models as judges is a great way to emulate ground truth.

It will work fastest loaded on a GPU (which we have plenty of at Modal), so we add a GPU to the modal.cls and write an @enter function to load the model into memory in advance.

```
Detector:        QReader
True Positives:  110
False Positives: 19
True Negatives:  0
False Negatives: 0
Score:           0.853
```

No way! Our QReader detected 100% of the QR codes, even the 19 codes that the iphone couldn't detect.

It's great we found such an impressive model, but we've overshot. Again, we don't want to find the perfect detector; we want to find the one that most closely matches iphone.

If we went to prod with this model, it would just say 100% pass, and we wouldn't learn anything useful for improving the model.

Thankfully, we can try something new. QReader being a Yolo model, it has an internal config for a `confidence threshold` which defaults to 50%. If we increase the threshold, we won't get so many positive detections.

I wrote a program to do a sweep of the different thresholds
`modal run backend.evals.scannability.optimize_qreader_threshold`
```
Testing threshold: 0.500
True Positives:  110
False Positives: 19
True Negatives:  0
False Negatives: 0
Score:           0.853

Testing threshold: 0.570
True Positives:  110
False Positives: 19
True Negatives:  0
False Negatives: 0
Score:           0.853

Testing threshold: 0.640
True Positives:  110
False Positives: 19
True Negatives:  0
False Negatives: 0
Score:           0.853

Testing threshold: 0.710
True Positives:  109
False Positives: 19
True Negatives:  0
False Negatives: 1
Score:           0.845

Testing threshold: 0.780
True Positives:  109
False Positives: 19
True Negatives:  0
False Negatives: 1
Score:           0.845

Testing threshold: 0.850
True Positives:  107
False Positives: 19
True Negatives:  0
False Negatives: 3
Score:           0.829

Testing threshold: 0.920
True Positives:  101
False Positives: 19
True Negatives:  0
False Negatives: 9
Score:           0.783

Testing threshold: 0.990
True Positives:  0
False Positives: 0
True Negatives:  19
False Negatives: 110
Score:           0.147
```

And we have a new problem! Increasing threshold did increase the amount of negative detections, but they were all false negatives! This means that QReader's ability to read difficult QR codes is on a different set of QRs than the iphone.

Sadly, optimizing threshold didn't give us a better detector.

### Ensemble

Now let's try a different method, where we run two detectors rather than just one and consider the QR as detected if at least one detector finds it (the OR operation)

This is called an ensemble.

And ensemble is only helpful if the two methods don't have much overlap, meaning one frequently detect positives when the other doesn't, and visa versa.

On all images, let's visually gut check their overlap.
```
OpenCV:
[1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Pyzbar:
[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```
Frequently, OpenCV and Pyzbar have different results for the same image, meaning we'd improve detection rate by taking the OR of the two (at risk of false positives).

`modal run backend.evals.scannability::test_ensemble`
yields
```
Detector:        Ensemble
True Positives:  37
False Positives: 1
True Negatives:  18
False Negatives: 73
Score:           0.426
```
Better!! Few false positives, more true positives than we've seen with the standalone detectors. 

But we still have plenty of false negatives, so let's try something new: `conditional ensembling`

This is where we only ensemble with another detector if the first detector returns negative. Note that this `conditional ensembling` is not some well-known evals term that you were too stupid to know. We literally invented it after looking at the above data asking: how do we make these numbers better? Again, evals are just code you write, no need to be intimidated by them.

### Finding the best ensemble
The opencv + pyzbar ensemble worked great, but if we add qreader into the mix again, there are too many combinations of thresholds we could use for us to easily logic through it. So let's calculate it.

I wrote a script that mapped out our different detectors (including the iphone) and their detections into binary. This includes qreader of various thresholds centered around the critical 0.92 confidence threshold that we'd previously seen be the tipping point in qreader.
`modal run backend.evals.scannability::generate_detection_table`
```
{
    "iphone":       [1, 1, 1, 1, 1, 1, ...],
    "opencv":       [1, 0, 0, 0, 1, 0, ...],
    "pyzbar":       [0, 0, 0, 0, 0, 1, ...],
    "qreader@0.86": [1, 1, 1, 1, 1, 1, ...],
    "qreader@0.88": [1, 1, 1, 1, 1, 1, ...],
    "qreader@0.9":  [1, 1, 1, 1, 1, 1, ...],
    "qreader@0.92": [1, 1, 1, 1, 1, 1, ...],
    "qreader@0.94": [1, 0, 1, 1, 1, 1, ...],
    "qreader@0.96": [0, 0, 0, 0, 0, 0, ...]
}
```

We then imagine different ensemble policies:
- opencv
- pyzbar 
- opencv + pyzbar ensemble
- qreader
- opencv with conditional fallback to qreader
- pyzbar with conditional fallback to qreader
- opencv + pyzbar ensemble with conditional fallback to qreader

Across multiple qreader thresholds, we calculate stats of their ensembles:
`modal run backend.evals.scannability::score_detection_table`
```
Top 5 policies by accuracy:
qreader@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall': 0.9727272727272728, 'f0.5':
0.8713355048859934, 'accuracy': 0.8294573643410853}
opencv_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
pyzbar_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
ensemble_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
qreader@0.88: {'tp': 106, 'tn': 0, 'fp': 19, 'fn': 4, 'precision': 0.848, 'recall': 0.9636363636363636, 'f0.5':
0.8688524590163934, 'accuracy': 0.8217054263565892}

Top 5 policies by precision:
opencv@0.86: {'tp': 21, 'tn': 19, 'fp': 0, 'fn': 89, 'precision': 1.0, 'recall': 0.19090909090909092, 'f0.5':
0.5412371134020618, 'accuracy': 0.31007751937984496}
opencv@0.88: {'tp': 21, 'tn': 19, 'fp': 0, 'fn': 89, 'precision': 1.0, 'recall': 0.19090909090909092, 'f0.5':
0.5412371134020618, 'accuracy': 0.31007751937984496}
opencv@0.9: {'tp': 21, 'tn': 19, 'fp': 0, 'fn': 89, 'precision': 1.0, 'recall': 0.19090909090909092, 'f0.5':
0.5412371134020618, 'accuracy': 0.31007751937984496}
opencv@0.92: {'tp': 21, 'tn': 19, 'fp': 0, 'fn': 89, 'precision': 1.0, 'recall': 0.19090909090909092, 'f0.5':
0.5412371134020618, 'accuracy': 0.31007751937984496}
opencv@0.94: {'tp': 21, 'tn': 19, 'fp': 0, 'fn': 89, 'precision': 1.0, 'recall': 0.19090909090909092, 'f0.5':
0.5412371134020618, 'accuracy': 0.31007751937984496}

Top 5 policies by recall:
qreader@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall': 0.9727272727272728, 'f0.5':
0.8713355048859934, 'accuracy': 0.8294573643410853}
opencv_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
pyzbar_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
ensemble_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
qreader@0.88: {'tp': 106, 'tn': 0, 'fp': 19, 'fn': 4, 'precision': 0.848, 'recall': 0.9636363636363636, 'f0.5':
0.8688524590163934, 'accuracy': 0.8217054263565892}

Top 5 policies by F0.5:
qreader@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall': 0.9727272727272728, 'f0.5':
0.8713355048859934, 'accuracy': 0.8294573643410853}
opencv_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
pyzbar_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
ensemble_w_qreader_backup@0.86: {'tp': 107, 'tn': 0, 'fp': 19, 'fn': 3, 'precision': 0.8492063492063492, 'recall':
0.9727272727272728, 'f0.5': 0.8713355048859934, 'accuracy': 0.8294573643410853}
qreader@0.88: {'tp': 106, 'tn': 0, 'fp': 19, 'fn': 4, 'precision': 0.848, 'recall': 0.9636363636363636, 'f0.5':
0.8688524590163934, 'accuracy': 0.8217054263565892}
```

Where Precision: "When we say a QR is scannable, how often are we right?"
Where Recall:    "Out of all scannable QRs, what portion do we correctly identify?"

Where F-beta Score = (1 + β²) * (precision * recall) / (β² * precision + recall) which combines precision and recall into a single score, weighted between precision and recall by β.

β = 1: F1 score, weights precision and recall equally
β = 0.5: F0.5 score, weights precision 2x more than recall

Unfortunately, this experiment didn't surface much novel information to us. Qreader at 0.86 threshold, standalone or conditionally ensembled, dominates most of the metrics. But looking at it, I'm still very dissatisfied with the quantity of false positives. We can make all the fancy F-beta scoring we want, but we know best.

But it's ok, experimentation is part of the process.

At this point, I was thinking maybe we should have just set up a screen with an iphone pointed at it.

But let's take our humble OpenCV + Pyzbar ensemble as our detector in our eval, and call it a day.

## Aesthetic Preference
Our "Ground Truth" for Aesthetic preference is much more subjective than iphone scanning. 

It's just: do you like the vibes of the image?

But while subjective, it's still important to maintain a consistent preference, so we can find a classifier for our evals that consistently matches that preference.


### Building a dataset
I write a script to reuse the 100+ images from the QR detection, because why waste good data.
 `modal run backend.evals.aesthetics::copy_scannability_set`.

It dumped them into `backend/evals/data/aesthetics`

### Sorting

Most of the scannability images were bad, a horrible mashup of randon concepts (as expected).

[bad image here]

I only found 9 of the 129 that felt even close to aesthetic. 

While manually sorting, I found that the things I considered most aesthetic often:
- Contained complete objects
- Were three dimensional in a way where the QR code wasn't just a flat rectangle, but built across objects at varying depths. 

Great example was one with a candle in the foreground that was the lower left anchor square of the QR

[image here]

or the glass of tea with a reflection on the glass that served as a data rectangle in the QR

[image here]

or a playground where the upper half of the QR was a house in the background

[image here]

### Improving the dataset

We need far more positive examples, so I tweaked the prompt to try to hit those same aesthetic traits that I liked.
```
prompt = f"A {adjective1} {noun1}, 3D render in blender, trending on artstation, uses depth to trick the eye, colorful, high resolution, 8k, cinematic, concept art, illustration, painting, modern"
```
`modal run backend.evals.aesthetics::generate_aesthetics_set_manual`

Using random adjectives and nouns resulted in 100% bad images, so I just scrapped them.

I refocused on getting the highest quality results possible, manually writing a dozen prompts and iterating them to higher and higher quality.
`modal run backend.evals.aesthetics::generate_aesthetics_set_random`

After a few iterations, we're getting even better results that made me bump previous "good" results into "bad" buckets since the new stuff is so good.

The new stuff isn't scannable, but this is for aesthetic preference, so it doesn't matter. In fact, it's great for us because it makes our eval overlap less with the scannability eval.

Continuing along that line of thought, since we're throwing scannability out the window, let's lower the `controlnet_conditioning_scale` on the model from 1.5 to 1.3. This resulted in a few very aesthetic images such as

[the artists_loft...v1 image]

Great. 

### Experimenting with Aesthetics rankers
Aesthetics is harder to programmatically test than qr scannability, so we'll lean heavily onto pretrained models from the community.

### Improved Aesthetics Predictor
https://huggingface.co/camenduru/improved-aesthetic-predictor

This model uses the CLIP image embedding model fed into a custom neaural net, and trained on the AVA (Aesthetic Visual Analysis) dataset.

The huggingface model wasn't usable out-of-the-box, so we had to rework it into our own library, found in `aesthetics.py`. We move the imports into the functions (to avoid importing locally), and then wrap the model in a [Modal class](link) ImprovedAestheticPredictor, so we can run it on GPUs.

The output of this model is a score between 0-10, with 10 being the most aesthetic. If the model is representative of our own aesthetic tastes, we'll be able to run our sorted good and bad images through it, and see a clear difference in the scores.

Let's do that! 

`modal run backend.evals.aesthetics::compare_aesthetics_predictions`

The above script scores our sorted images, and then plots them by good/bad class into this histogram:
[aesthetics_scores.png here]

We love to see a graph like this. Specifically:
1. The data for the good and bad buckets each form a bell curve, meaning our aesthetic preference model is actually doing something. (todo: this feels like a dumb statement that'd get us flamed by data folks, needs editing)
2. The bell curves are offset from each other, with the bulk of the good images scoring higher than the bad images. This is a clear sign that our model aligns with our aesthetic preferences.

Note that the graph isn't perfect, and there's plenty of overlap between the good and bad buckets. 

In an ideal world, with no overlap, we could just select a threshold between the two, and have no false predictions; Anything above threshould would be a true positive, and anything below threshold would be a true negative.

But in our case, we do have overlap, so choosing a threshold in that overlap will introduce false positives and negatives. Choose high, you'd reduce false positives (fewer bad images fall above the threshold), but you'd increase false negatives (more good images fall below the threshold). Choose low, you'd increase false positives, but you'd reduce false negatives.

 Thankfully, we can choose what matters most to our QR Code usecase. As we continue to iterate on our model, we want our QRs to become more aesthetically pleasing, so it doesn't hurt to have a high bar.

 Eyeballing the graph, let's choose a threshold of 6.0. The majority of good images fall above this threshold, and fewer than 15% of bad images do.

 ## Putting it all together

 Now, we finally get to write the code that runs the evals on newly generated images, and evaluates the performance of those generations.

 This is the holy script. The test we'll run on every iteration of the model and inference codebase in order to measure if we're progressing or regressing our product.

 Remember: Models are temporary. Evals are forever.
 [insert charles photo here]

In `evals/evals.py`, we write a `run_evals` function that does the following:
1. Chooses a quantity of tests to run (N_TESTS). A test is a unique prompt + QR code combo.
2. Chooses a quantity of images to generate per test (N_SAMPLES). We generate multiple images for each test so we can calculate pass@k metrics.
3. Generates N_TESTS * N_SAMPLES images in parallel, using the `generate_image` function we'd previously written.
4. Groups those images into N_TESTS batches, and runs the `run_aesthetics_eval` and `run_scannability_eval` functions on each batch, which use `detect_qr_ensemble` and `ImprovedAestheticPredictor`. This returns a dictionary of pass@k metrics for each test.
5. Averages the pass@k metrics across all tests.

For example, if we set N_SAMPLES=3, the eval could output logs like:
```
Scannability Test 1:
Passed: [1, 0, 0]
Samples passed: 1/3
Estimated pass@1: 33.33%
Estimated pass@3: 100.00%
Estimated pass@10: 100.00%
```
For a single test (of many), we'd see that of three samples generated from the same prompt / QR combo, one passed.

So for pass@1, if we were looking at just one image, 33% chance that we select the one that passed the scannability test.

And for pass@3, if we were looking at three images, 100% chance that one of them passed the scannability test.

For pass@10, we didn't generate 10 images, but since one of just three passed, that's a 100% chance that one of ten would pass.

This calculation is made across all tests, and then averaged into our final score.

## Experiment tracking

Evals are a form of exploration and experimentation. Yet many teams learn the following lesson the hard way:

Track your evals. Track the exact code and model used to get those results.

Since you're going to be iterating quickly on your model and inference codebase, you'l want to be able to easily backtrack to your highest performing version. 

I've seen first-hand the pains of getting incredible evals, and failing to reproduce them. You don't want this.

We use [Weights & Biases](link) to track our eval runs. Weights and Biases is a tool for experiment tracking, and is integrated with very few lines of code. With it comes:

A dashboard to visualize your runs
[insert screenshot here]

And a code-saving tool that takes your code in the exact state it was at the time of the run and saves it as an artifact on the run. (todo: make this work, it's currently broken)

Combined, you now have the tools to fearlessly experiment on new models, new prompts, new sampling code, anything inference related, and easily backtrack to your best performing version.

## Conclusion

(todo: write this)