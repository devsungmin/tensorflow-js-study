//express, ejs 설치 해줄것
var express = require('express')
var app = express()

app.set('view engine', 'ejs')
app.set('views', './') // views = './views'
app.engine('html', require('ejs').renderFile)
app.use(express.static(__dirname))

app.get('/', function(req, res){
    res.render("index.html")
})

app.listen(3000, function(){
    console.log("listening on 3000")})