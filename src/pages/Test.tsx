import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { Col, Container, Form, Row, Spinner } from "react-bootstrap"
;
function Test() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [model, setModel] = useState<tf.GraphModel<string | tf.io.IOHandler>>();
  const imageRef = useRef<any>(null);
  const [isNewOdModel, setIsNewOdModel] = useState<boolean>();
  const NEW_OD_OUTPUT_TENSORS = [
    "detected_boxes",
    "detected_scores",
    "detected_classes",
  ];
  const ANCHORS = [
    0.573, 0.677, 1.87, 2.06, 3.34, 5.47, 7.88, 3.53, 9.77, 9.17,
  ];

  useEffect(() => {
    loadCustomModel();
  }, []);

  const loadCustomModel = async () => {
    setIsLoading(true);
    const model = await tf.loadGraphModel(
      "https://azure-tf.b-cdn.net/apple-tensor-flow/model.json",
      { onProgress: showProgress }
    );
    setIsNewOdModel(model.inputs.length === 3);
    setModel(model);
  };

  const showProgress = (percentage: number) => {
    var pct = Math.floor(percentage * 100.0);
    console.log("pct", pct);

    if (pct === 100) {
      setIsLoading(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      let reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = function () {
        imageRef.current.src = reader.result;
      };
      reader.onerror = function (error) {
        console.log("Error: ", error);
      };
    }
  };

  const loadImage = async () => {
    // Pre-process the image
    const input_size: any =
      model &&
      model.inputs[0] &&
      model.inputs[0].shape &&
      model.inputs[0].shape[2] &&
      model.inputs[0].shape[2];
    console.log("input_size", input_size);
    console.log("imageRef.current", imageRef.current);

    let image: any = tf.browser.fromPixels(imageRef.current, 3);
    console.log("imageimage", image);

    image = tf.image.resizeBilinear(image.expandDims().toFloat(), [
      input_size,
      input_size,
    ]);
    if (isNewOdModel) {
      console.log("Object Detection Model V2 detected.");
      image = isNewOdModel ? image : image.reverse(-1); // RGB->BGR for old models
    }
    console.log("image", image);

    return image;
  };

  const predictLogos = async (image: any) => {
    console.log("Running predictions...", image, isNewOdModel);
    if (model) {
      console.log("imageimageimageimageimage", image);

      const outputs = await model.executeAsync(
        image,
        isNewOdModel ? NEW_OD_OUTPUT_TENSORS : undefined
      );

      console.log("outputs", outputs);

      const arrays = !Array.isArray(outputs)
        ? await outputs.array()
        : Promise.all(outputs.map((t) => t.array()));
      let predictions: any = await arrays;
      console.log("arrays", predictions);

      // Post processing for old models.
      if (predictions.length != 3) {
        console.log("Post processing...");
        const num_anchor = ANCHORS.length / 2;
        const channels = predictions[0][0][0].length;
        const height = predictions[0].length;
        const width = predictions[0][0].length;
        console.log("num_anchor", num_anchor);
        console.log("channels", channels);
        console.log("height", height);
        console.log("width", width);

        const num_class = channels / num_anchor - 5;
        console.log("num_class", num_class);

        let boxes: any = [];
        let scores: any = [];
        let classes: any = [];

        for (var grid_y = 0; grid_y < height; grid_y++) {
          for (var grid_x = 0; grid_x < width; grid_x++) {
            let offset = 0;

            for (var i = 0; i < num_anchor; i++) {
              let x =
                (_logistic(predictions[0][grid_y][grid_x][offset++]) + grid_x) /
                width;
              let y =
                (_logistic(predictions[0][grid_y][grid_x][offset++]) + grid_y) /
                height;
              let w =
                (Math.exp(predictions[0][grid_y][grid_x][offset++]) *
                  ANCHORS[i * 2]) /
                width;
              let h =
                (Math.exp(predictions[0][grid_y][grid_x][offset++]) *
                  ANCHORS[i * 2 + 1]) /
                height;

              let objectness = tf.scalar(
                _logistic(predictions[0][grid_y][grid_x][offset++])
              );
              let class_probabilities = tf
                .tensor1d(
                  predictions[0][grid_y][grid_x].slice(
                    offset,
                    offset + num_class
                  )
                )
                .softmax();
              offset += num_class;

              class_probabilities = class_probabilities.mul(objectness);
              let max_index = class_probabilities.argMax();
              boxes.push([x - w / 2, y - h / 2, x + w / 2, y + h / 2]);
              scores.push(class_probabilities.max().dataSync()[0]);
              classes.push(max_index.dataSync()[0]);
            }
          }
        }

        boxes = tf.tensor2d(boxes);
        scores = tf.tensor1d(scores);
        classes = tf.tensor1d(classes);
        console.log("boxes", boxes);
        console.log("scores", scores);
        console.log("classes", classes);

        const selected_indices = await tf.image.nonMaxSuppressionAsync(
          boxes,
          scores,
          10
        );
        console.log("selected_indices", selected_indices);

        predictions = [
          await boxes.gather(selected_indices).array(),
          await scores.gather(selected_indices).array(),
          await classes.gather(selected_indices).array(),
        ];
      }
      console.log("predictions[0] :)", predictions[0]);
      return predictions;
    }
  };

  const highlightResults = async (predictions: any) => {
    console.log("Highlighting results...");

    // Assuming you have defined TARGET_CLASSES and selectedImage somewhere in your code
    const TARGET_CLASSES = ["Class1", "Class2", "Class3"]; // Replace with your actual class names
    const selectedImage = imageRef.current;
    const imageOverlay = document.getElementById("imageOverlay");

    console.log("imageOverlay", imageOverlay);
    // Assuming you have defined bboxLeft, bboxTop, bboxWidth, and bboxHeight
    let bboxLeft, bboxTop, bboxWidth, bboxHeight;

    // Remove existing highlights

    for (let n = 0; n < predictions[0].length; n++) {
      console.log("predictions[1][n]", predictions[1][n]);

      // Check scores
      if (predictions[1][n] > 0.66) {
        console.log("predictions[1][n] > 0.66", predictions[1][n]);
        const p = document.createElement("p");
        p.innerText =
          TARGET_CLASSES[predictions[2][n]] +
          ": " +
          Math.round(parseFloat(predictions[1][n]) * 100) +
          "%";

        bboxLeft = predictions[0][n][0] * selectedImage.width + 10;
        bboxTop = predictions[0][n][1] * selectedImage.height - 10;
        bboxWidth = predictions[0][n][2] * selectedImage.width - bboxLeft + 20;
        bboxHeight = predictions[0][n][3] * selectedImage.height - bboxTop + 10;

        p.style.setProperty("margin-left", bboxLeft + "px");
        p.style.setProperty("margin-top", bboxTop - 10 + "px");
        p.style.setProperty("width", bboxWidth + "px");
        p.style.setProperty("top", "0");
        p.style.setProperty("left", "0");
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style.setProperty("left", bboxLeft + "px");
        highlighter.style.setProperty("top", bboxTop + "px");
        highlighter.style.setProperty("width", bboxWidth + "px");
        highlighter.style.setProperty("height", bboxHeight + "px");
        if (imageOverlay) {
          imageOverlay.appendChild(highlighter);
          imageOverlay.appendChild(p);
        }
      }
    }
  };

  const _logistic = (x: number) => {
    if (x > 0) {
      return 1 / (1 + Math.exp(-x));
    } else {
      const e = Math.exp(x);
      return e / (1 + e);
    }
  };

  const Predict = async () => {
    const image = await loadImage();
    console.log("image FINAL", image);
    const predict = await predictLogos(image);
    console.log("predict FINAL", predict);
    await highlightResults(predict);
  };

  return (
    <div>
      {isLoading ? (
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      ) : null}
      {!isLoading && (
        <Container>
          <Row sm={4}>
            <Col sm={4}>
              <Form.Group controlId="formFile" className="mb-3 mt-3">
                <Form.Label>Select Image</Form.Label>
                <Form.Control type="file" onChange={handleImageUpload} />
              </Form.Group>
              {/* <img
                width="500"
                height="500"
                ref={imageRef}
                src={
                  imageRef.current && imageRef.current.src
                    ? imageRef.current.src
                    : ""
                }
                alt="Uploaded"
              /> */}
              <div className="row">
                <div className="col-12">
                  <h2 className="ml-3">Image</h2>
                  <div className="col-6">
                    <button
                      onClick={() => Predict()}
                      id="predict-button"
                      className="btn btn-primary float-right"
                    >
                      Predict
                    </button>
                  </div>
                  <div id="imageOverlay" className="imageOverlay">
                    <img
                      ref={imageRef}
                      id="selectedImage"
                      className="ml-3"
                      width="500"
                      height="auto"
                      alt=""
                      src={
                        imageRef.current && imageRef.current.src
                          ? imageRef.current.src
                          : ""
                      }
                    />
                  </div>
                </div>
              </div>
            </Col>
          </Row>
        </Container>
      )}
    </div>
  );
}

export default Test;
