import json

def parse_json(js_path):
    annotation = dict()
    with open(js_path) as json_file:
        data=json.load(json_file)
        print(data)
        img_name = data['filename']
        size = data['size']
        h=size[0]
        w=size[1]
        bboxes=data['bboxes']
        annotation = {
            'filename': img_name,
            'width': w,
            'height': h,
            'bboxes': bboxes.astype(np.float32)
        }
    return annotation

print(parse_json('./ex.json'))

# def parse_xml(args):
#     xml_path, img_path = args
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)
#     bboxes = []
#     labels = []
#     bboxes_ignore = []
#     labels_ignore = []
#     for obj in root.findall('object'):
#         name = obj.find('name').text
#         label = label_ids[name]
#         difficult = int(obj.find('difficult').text)
#         bnd_box = obj.find('bndbox')
#         bbox = [
#             int(bnd_box.find('xmin').text),
#             int(bnd_box.find('ymin').text),
#             int(bnd_box.find('xmax').text),
#             int(bnd_box.find('ymax').text)
#         ]
#         if difficult:
#             bboxes_ignore.append(bbox)
#             labels_ignore.append(label)
#         else:
#             bboxes.append(bbox)
#             labels.append(label)
#     if not bboxes:
#         bboxes = np.zeros((0, 4))
#         labels = np.zeros((0, ))
#     else:
#         bboxes = np.array(bboxes, ndmin=2) - 1
#         labels = np.array(labels)
#     if not bboxes_ignore:
#         bboxes_ignore = np.zeros((0, 4))
#         labels_ignore = np.zeros((0, ))
#     else:
#         bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
#         labels_ignore = np.array(labels_ignore)
#     annotation = {
#         'filename': img_path,
#         'width': w,
#         'height': h,
#         'ann': {
#             'bboxes': bboxes.astype(np.float32),
#             'labels': labels.astype(np.int64),
#             'bboxes_ignore': bboxes_ignore.astype(np.float32),
#             'labels_ignore': labels_ignore.astype(np.int64)
#         }
#     }
#     return annotation