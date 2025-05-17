module.exports = {
  transpileDependencies: ["vuetify"],
  lintOnSave: false, // ← add this line
  devServer: {
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        ws: true,
        changeOrigin: true,
        pathRewrite: { "^/api": "" },
      },
    },
  },
};
