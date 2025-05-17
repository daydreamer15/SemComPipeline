<template>
  <v-app>
    <v-app-bar app color="primary" dark>
      <v-toolbar-title
        >ESemCom: Evolving Semantic Communication with Generative
        Model</v-toolbar-title
      >
    </v-app-bar>

    <v-main>
      <router-view />
    </v-main>

    <v-footer app>
      <span>&copy; {{ new Date().getFullYear() }} - ESemCom Demo</span>
    </v-footer>
  </v-app>
</template>

<script>
export default {
  name: "App",
  data: () => ({
    //
  }),
};
</script>

<style>
.image-preview {
  max-width: 100%;
  max-height: 300px;
}
</style>
