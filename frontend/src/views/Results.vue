<template>
  <v-container fluid>
    <v-row justify="center">
      <v-col cols="12">
        <v-card class="mb-5">
          <v-card-title class="text-h5"
            >ESemCom Processing Pipeline</v-card-title
          >
          <v-card-subtitle
            >Follow the image processing through the semantic communication
            system</v-card-subtitle
          >

          <v-card-text v-if="results">
            <!-- The pipeline visualization with arrows -->
            <div class="pipeline-container">
              <div
                v-for="(stage, index) in stages"
                :key="index"
                class="pipeline-stage"
              >
                <div class="stage-box">
                  <h3 class="stage-title">{{ stage.title }}</h3>
                  <v-img
                    :src="`data:image/png;base64,${results[stage.key]}`"
                    max-height="150"
                    contain
                    class="mx-auto"
                  ></v-img>
                </div>
                <v-icon
                  v-if="index < stages.length - 1"
                  class="pipeline-arrow"
                  large
                >
                  mdi-arrow-right
                </v-icon>
              </div>
            </div>

            <!-- Metrics Display -->
            <v-card class="mt-5" outlined>
              <v-card-title>Performance Metrics</v-card-title>
              <v-card-text>
                <v-row>
                  <v-col cols="12" md="4">
                    <v-card outlined>
                      <v-card-title
                        >BCR (Bandwidth Compression Ratio)</v-card-title
                      >
                      <v-card-text class="text-h5 text-center">
                        1/{{ Math.round(1 / results.metrics.bcr) }}
                      </v-card-text>
                    </v-card>
                  </v-col>

                  <v-col cols="12" md="4">
                    <v-card outlined>
                      <v-card-title
                        >PSNR (Peak Signal-to-Noise Ratio)</v-card-title
                      >
                      <v-card-text class="text-h5 text-center">
                        {{ results.metrics.psnr }} dB
                      </v-card-text>
                    </v-card>
                  </v-col>

                  <v-col cols="12" md="4">
                    <v-card outlined>
                      <v-card-title
                        >LPIPS (Learned Perceptual Image
                        Similarity)</v-card-title
                      >
                      <v-card-text class="text-h5 text-center">
                        {{ results.metrics.lpips }}
                      </v-card-text>
                    </v-card>
                  </v-col>
                </v-row>
              </v-card-text>
            </v-card>
          </v-card-text>

          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn color="primary" @click="goHome"> Try Another Image </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: "Results",
  data() {
    return {
      results: null,
      stages: [
        { title: "Original Image", key: "original" },
        { title: "Semantic Encoder", key: "encoded" },
        { title: "Transmitter Cache", key: "transmitter_cached" },
        { title: "Power Normalization", key: "power_normalized" },
        { title: "Noisy Channel", key: "noisy_channel" },
        { title: "Receiver Cache", key: "receiver_cached" },
        { title: "Semantic Decoder", key: "reconstructed" },
      ],
    };
  },
  created() {
    if (this.$route.params.results) {
      this.results = this.$route.params.results;
    } else {
      this.goHome();
    }
  },
  methods: {
    goHome() {
      this.$router.push({ name: "Home" });
    },
  },
};
</script>

<style scoped>
.pipeline-container {
  display: flex;
  flex-wrap: nowrap;
  overflow-x: auto;
  padding: 20px 10px;
  align-items: center;
}

.pipeline-stage {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.stage-box {
  width: 180px;
  height: 220px;
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  background-color: #f9f9f9;
}

.stage-title {
  text-align: center;
  font-size: 14px;
  margin-bottom: 10px;
}

.pipeline-arrow {
  margin: 0 10px;
}
</style>
