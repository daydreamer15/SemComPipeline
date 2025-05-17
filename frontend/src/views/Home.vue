<template>
  <v-container fluid>
    <v-row justify="center" align="center">
      <v-col cols="12" md="8">
        <v-card class="mx-auto mt-5" max-width="800">
          <v-card-title class="text-h4 justify-center">
            Evolving Semantic Communication
          </v-card-title>

          <v-card-text>
            <p class="text-body-1">
              This demo illustrates the ESemCom system described in the paper
              "Evolving Semantic Communication with Generative Model". Upload an
              image to see how it's processed through the semantic communication
              pipeline.
            </p>

            <v-divider class="my-5"></v-divider>

            <v-row justify="center" class="my-5">
              <v-btn
                color="primary"
                @click="triggerUpload"
                :loading="loading"
                :disabled="loading"
              >
                Choose Image
              </v-btn>
              <input
                type="file"
                ref="fileInput"
                style="display: none"
                @change="onFileSelected"
                accept="image/*"
              />
            </v-row>

            <v-alert v-if="error" type="error" dismissible class="mt-3">
              {{ error }}
            </v-alert>

            <v-row v-if="selectedFile" justify="center" class="mt-5">
              <v-col cols="12" class="text-center">
                <h3>Selected Image</h3>
                <v-img
                  :src="previewUrl"
                  max-height="300"
                  contain
                  class="mx-auto"
                ></v-img>
                <v-btn
                  color="success"
                  class="mt-5"
                  @click="processImage"
                  :loading="loading"
                  :disabled="loading"
                >
                  Process Image
                </v-btn>
              </v-col>
            </v-row>

            <v-row v-if="selectedFile" justify="center" class="mt-3">
              <v-col cols="12" sm="6">
                <v-slider
                  v-model="snr"
                  :tick-labels="['Low', 'Medium', 'High']"
                  :max="2"
                  step="1"
                  ticks="always"
                  tick-size="4"
                  label="Channel Quality (SNR)"
                ></v-slider>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import axios from "axios";

export default {
  name: "Home",
  data() {
    return {
      selectedFile: null,
      previewUrl: null,
      loading: false,
      error: null,
      snr: 1, // Default to medium SNR
    };
  },
  methods: {
    triggerUpload() {
      this.$refs.fileInput.click();
    },
    onFileSelected(event) {
      this.selectedFile = event.target.files[0];
      if (!this.selectedFile) return;

      // Create preview
      this.previewUrl = URL.createObjectURL(this.selectedFile);
    },
    processImage() {
      if (!this.selectedFile) {
        this.error = "Please select an image first";
        return;
      }

      this.loading = true;
      this.error = null;

      // First set the SNR value
      const snrValues = [0, 5, 10]; // Low, Medium, High
      axios
        .post("/api/set_snr", { snr: snrValues[this.snr] })
        .then(() => {
          // Now upload the image
          const formData = new FormData();
          formData.append("file", this.selectedFile);

          return axios.post("/api/upload", formData, {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          });
        })
        .then((response) => {
          this.$router.push({
            name: "Results",
            params: {
              results: response.data,
            },
          });
        })
        .catch((err) => {
          console.error("Error processing image:", err);
          this.error = "Error processing image. Please try again.";
        })
        .finally(() => {
          this.loading = false;
        });
    },
  },
};
</script>
