const express = require('express');
const app = express();
const port = 3000;

let ledState = {
    led1: false,
    led2: false,
    led3: false,
    led4: false
};

app.use(express.json());
app.use(express.static('public'));

app.get('/status', (req, res) => {
    res.json(ledState);
    console.log('Status requested:', ledState);
});

app.post('/control', (req, res) => {
    const { led } = req.body;
    if (ledState.hasOwnProperty(led)) {
        // 切换LED状态
        ledState[led] = !ledState[led];
        res.json(ledState);
        console.log(`LED ${led} state changed to:`, ledState[led]);
    } else {
        res.status(400).send('Invalid LED');
        console.log('Invalid LED:', led);
    }
});

app.listen(port, () => {
    console.log(`Virtual LED server running at http://localhost:${port}`);
});
