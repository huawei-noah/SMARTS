const HtmlWebPackPlugin = require("html-webpack-plugin");

module.exports = {
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: { loader: "babel-loader" }
            },
            {
                test: /\.html$/,
                use: [{ loader: "html-loader" }]
            },
            {
                test: /\.css$/,
                use: [{ loader: "style-loader" }, { loader: "css-loader" }]
            },
            {
                test: /\.m?js$/,
                resolve: {
                    fullySpecified: false, // disable the behaviour (see https://github.com/webpack/webpack/issues/11467)
                },
            },
        ]
    },
    devServer: {
        historyApiFallback: true,
    },
    devtool: "source-map",
    plugins: [
        new HtmlWebPackPlugin({
            template: "./src/index.html",
            filename: "./index.html",
            title: "Envision",
            showErrors: true,
        })
    ]
};
