const path = require('path');
const portfinder = require('portfinder');
const TerserPlugin = require("terser-webpack-plugin");

const config = {
  entry: ['./src/main.js'],
  output: {
    filename: 'webnn-polyfill.js',
    path: path.resolve(__dirname, 'dist')
  },
	module: {
		rules: [
      {
        test: /\.js$/,
        loader: 'babel-loader',
        exclude: /node_modules/
      },
      {
        test: /\.wasm$/i,
        type: 'javascript/auto',
        loader: 'file-loader',
        options: {
          name: '[name].[ext]'
        }
      },
      {
        test: /\.tsx?$/,
        loader: 'ts-loader',
        exclude: /node_modules/,
        options: {
          configFile: require.resolve('./tsconfig.json'),
          compilerOptions: { noEmit: false }
        }
      }
    ]
  },
  resolve: {
		extensions: ['.js', '.tsx', '.ts']
  },
  devServer: {
    // enable https
    https: process.env.HTTPS === 'true' || false,
    // allow connections from LAN
    host: '0.0.0.0',
    // allow connections using hostname
    disableHostCheck: true,
    // serve bundle files from /dist/ without writing to disk
    publicPath: '/dist/',
  },
  optimization: {
    minimize: true,
    minimizer: [
      new TerserPlugin({
        extractComments: false,
        terserOptions: {
          compress: {
            typeofs: false,
          },
        },
      }),
    ],
  },
};

if (process.env.NODE_ENV === 'production') {
  config.mode = 'production';
  // generate a separate source map file
  // exclude it from production server if you don't want to enable source map
  config.devtool = 'source-map';
} else {
  config.mode = 'development';
  // inline the source map in bundle file for remote debugging
  config.devtool = 'inline-source-map';
}

module.exports = new Promise((resolve) => {
  const basePort = 8080;
  portfinder.getPort({
    port: basePort
  }, (_, port) => {
    config.devServer.port = port;
    config.devServer.public = `localhost:${port}`;
    resolve(config);
  });
});
