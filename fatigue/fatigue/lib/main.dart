import 'package:flutter/material.dart';

class MyActivityScreen extends StatefulWidget {
  @override
  _MyActivityScreenState createState() => _MyActivityScreenState();
}

class _MyActivityScreenState extends State<MyActivityScreen> {
  TextEditingController heartRateController = TextEditingController();
  TextEditingController restingHeartRateController = TextEditingController();
  TextEditingController bmiController = TextEditingController();
  TextEditingController caloriesController = TextEditingController();
  String fileStatus = 'File not inserted';

  @override
  void dispose() {
    heartRateController.dispose();
    restingHeartRateController.dispose();
    bmiController.dispose();
    caloriesController.dispose();
    super.dispose();
  }

  void generateFasScore() {
    // Implement your logic for generating FAS score here
    // You can retrieve the values from the text fields using the respective controllers
  }

  void exportFile() {
    // Implement your logic for exporting the file here
    // You can check the file format and update the fileStatus accordingly
    setState(() {
      fileStatus = 'File inserted';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('My Activity Screen'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: heartRateController,
              decoration: InputDecoration(labelText: 'Heart Rate'),
            ),
            TextField(
              controller: restingHeartRateController,
              decoration: InputDecoration(labelText: 'Resting Heart Rate'),
            ),
            TextField(
              controller: bmiController,
              decoration: InputDecoration(labelText: 'BMI'),
            ),
            TextField(
              controller: caloriesController,
              decoration: InputDecoration(labelText: 'Calories'),
            ),
            SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: generateFasScore,
              child: Text('Generate FAS Score'),
            ),
            SizedBox(height: 32.0),
            ElevatedButton(
              onPressed: exportFile,
              child: Text('Export File'),
            ),
            SizedBox(height: 16.0),
            Text('File Status: $fileStatus'),
          ],
        ),
      ),
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: MyActivityScreen(),
  ));
}
