import cv2
import numpy as np

# Load model files
prototxt = "models/colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"
points = "models/pts_in_hull (1).npy"

# Load network
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# Add cluster centers as 1x1 convolution
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load grayscale image
image = cv2.imread("input.jpg")
scaled = image.astype("float32") / 255.0

# Convert to LAB color space
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize image
resized = cv2.resize(lab, (224, 224))
L = resized[:, :, 0]

# Normalize
L -= 50

# Predict color
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize back
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Combine with L channel
L = lab[:, :, 0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Save output
cv2.imwrite("output.png", (255 * colorized).astype("uint8"))

print("✅ Image colorized and saved as output.png")