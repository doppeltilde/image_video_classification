import 'dart:typed_data';

import 'package:http/http.dart' as http;

Map<String, String> headers = {
  "Content-Type": 'multipart/form-data',
  "Accept": 'application/json',
  "X-API-Key": "example",
};

Map<String, dynamic> queryParamters = {
  "model_names": [
    "LukeJacob2023/nsfw-image-detector",
    "rizvandwiki/gender-classification-2",
    "nateraw/vit-age-classifier",
  ],
  "labels": [
    "sexy",
    "hentai",
    "porn",
    "female",
    "male",
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39"
  ],
  "score": "0.1",
  "fast_mode": "true",
  "skip_frames_percentage": "5",
  "return_on_first_matching_label": "true",
};

String queryString = Uri(queryParameters: queryParamters).query;

var url = Uri.parse(
    'http://localhost:8000/api/image-query-classification?$queryString');

Future<void> main() async {
  final v = await fetchNetworkImage();

  var request = http.MultipartRequest(
    "POST",
    url,
  )
    ..headers.addAll(headers)
    ..files.add(http.MultipartFile.fromBytes(
      'file',
      v,
      filename: "example",
    ));

  final response = await http.Response.fromStream(await request.send());

  if (response.statusCode == 200) {
    print("==============================================");
    print("\n\n${response.body}\n\n");
    print("==============================================");
  }
}

Future<Uint8List> fetchNetworkImage() async {
  final examplePic = Uri.parse(
      "https://ia601507.us.archive.org/35/items/alla3.11.22igs/alla3.11.22igs1.jpg");
  print(url);

  final response = await http.get(examplePic);

  return response.bodyBytes;
}
