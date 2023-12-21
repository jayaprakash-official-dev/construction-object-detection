import React from "react";
import { Col, Container, Form, Row, Image } from "react-bootstrap";

function ImageObjectDetection() {
  return (
    <Container>
      <Row className="d-flex justify-content-between">
        <p className="fs-2 fw-bold text-left p-2">
          Azure CV + TensorFlow AI/ML Object Detection
        </p>
        <Col sm={4}>
          <Form.Label htmlFor="inputPassword5">Image URL</Form.Label>
          <Form.Control
            type="password"
            id="inputPassword5"
            aria-describedby="passwordHelpBlock"
          />
          <p className="text-start fs-5 fw-bold mt-2">OR</p>
          <Form.Group controlId="formFile" className="mb-3 mt-3">
            <Form.Label>Select Image</Form.Label>
            <Form.Control type="file" />
          </Form.Group>
          <p className="text-start fs-4 fw-bold">Image Preview</p>
          <Image
            fluid
            src="https://constructionready.org/wp-content/themes/yootheme/cache/female-construction-worker-min-5b669682.jpeg"
            rounded
          />
        </Col>
        <Col sm={6}>
          <div className="pl-3">
            <p className="text-start fs-4 fw-bold ">Result :</p>
            <Image
              fluid
              // style={{ width: 300 }}
              src="https://constructionready.org/wp-content/themes/yootheme/cache/female-construction-worker-min-5b669682.jpeg"
              rounded
            />
          </div>
        </Col>
      </Row>
    </Container>
  );
}

export default ImageObjectDetection;
