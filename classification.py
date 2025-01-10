import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import pandas as pd
from streamlit_extras.add_vertical_space import add_vertical_space
import time
model = load_model('model.h5')
labels = {
    0: 'Asparagus', 1: 'Crimson Clover', 2: 'Daisy Fleabane', 3: 'Fireweed', 4: 'Milk Thistle', 5: 'Sunflower',
    6: 'apple', 7: 'banana', 8: 'beetroot', 9: 'bell pepper', 10: 'cabbage', 11: 'capsicum', 12: 'carrot', 
    13: 'cauliflower', 14: 'chilli pepper', 15: 'corn', 16: 'cucumber', 17: 'eggplant', 18: 'garlic', 
    19: 'ginger', 20: 'grapes', 21: 'jalepeno', 22: 'kiwi', 23: 'lemon', 24: 'lettuce', 25: 'mango', 
    26: 'onion', 27: 'orange', 28: 'paprika', 29: 'pear', 30: 'peas', 31: 'pineapple', 32: 'pomegranate', 
    33: 'potato', 34: 'raddish', 35: 'soy beans', 36: 'spinach', 37: 'sweetcorn', 38: 'sweetpotato', 
    39: 'tomato', 40: 'turnip', 41: 'watermelon'
}

nutritional_values = {
    'Asparagus': {'Calories': 20, 'Protein (g)': 2.2, 'Carbs (g)': 3.9, 'Fiber (g)': 2.1, 'Vitamin C (mg)': 5.6},
    'Crimson clover': {'Calories': 25, 'Protein (g)': 2.6, 'Carbs (g)': 4.2, 'Fiber (g)': 1.8, 'Vitamin C (mg)': 10},
    'Daisy fleabane': {'Calories': 15, 'Protein (g)': 1.0, 'Carbs (g)': 2.8, 'Fiber (g)': 1.5, 'Vitamin C (mg)': 7},
    'Fireweed': {'Calories': 29, 'Protein (g)': 3.0, 'Carbs (g)': 5.1, 'Fiber (g)': 2.2, 'Vitamin C (mg)': 15},
    'Milk thistle': {'Calories': 40, 'Protein (g)': 3.6, 'Carbs (g)': 6.5, 'Fiber (g)': 2.9, 'Vitamin C (mg)': 12},
    'Sunflower': {'Calories': 584, 'Protein (g)': 20.8, 'Carbs (g)': 11.4, 'Fiber (g)': 8.6, 'Vitamin E (mg)': 35.2},
    'Apple': {'Calories': 52, 'Protein (g)': 0.3, 'Carbs (g)': 14, 'Fiber (g)': 2.4, 'Vitamin C (mg)': 4.6},
    'Banana': {'Calories': 96, 'Protein (g)': 1.3, 'Carbs (g)': 27, 'Fiber (g)': 2.6, 'Vitamin C (mg)': 8.7},
    'Beetroot': {'Calories': 43, 'Protein (g)': 1.6, 'Carbs (g)': 10, 'Fiber (g)': 2.8, 'Vitamin C (mg)': 4.9},
    'Bell pepper': {'Calories': 31, 'Protein (g)': 1.0, 'Carbs (g)': 6, 'Fiber (g)': 2.1, 'Vitamin C (mg)': 127},
    'Cabbage': {'Calories': 25, 'Protein (g)': 1.3, 'Carbs (g)': 6, 'Fiber (g)': 2.5, 'Vitamin C (mg)': 36.6},
    'Capsicum': {'Calories': 40, 'Protein (g)': 1.9, 'Carbs (g)': 9, 'Fiber (g)': 1.7, 'Vitamin C (mg)': 213},
    'Carrot': {'Calories': 41, 'Protein (g)': 0.9, 'Carbs (g)': 10, 'Fiber (g)': 2.8, 'Vitamin A (μg)': 835},
    'Cauliflower': {'Calories': 25, 'Protein (g)': 1.9, 'Carbs (g)': 5, 'Fiber (g)': 2, 'Vitamin C (mg)': 48.2},
    'Chilli pepper': {'Calories': 40, 'Protein (g)': 2.0, 'Carbs (g)': 9, 'Fiber (g)': 1.5, 'Vitamin C (mg)': 143},
    'Corn': {'Calories': 86, 'Protein (g)': 3.2, 'Carbs (g)': 19, 'Fiber (g)': 2.7, 'Vitamin C (mg)': 6.8},
    'Cucumber': {'Calories': 16, 'Protein (g)': 0.7, 'Carbs (g)': 3.6, 'Fiber (g)': 0.5, 'Vitamin C (mg)': 2.8},
    'Eggplant': {'Calories': 25, 'Protein (g)': 1.0, 'Carbs (g)': 6, 'Fiber (g)': 3, 'Vitamin C (mg)': 2.2},
    'Garlic': {'Calories': 149, 'Protein (g)': 6.4, 'Carbs (g)': 33, 'Fiber (g)': 2.1, 'Vitamin C (mg)': 31},
    'Ginger': {'Calories': 80, 'Protein (g)': 1.8, 'Carbs (g)': 18, 'Fiber (g)': 2.0, 'Vitamin C (mg)': 5},
    'Grapes': {'Calories': 69, 'Protein (g)': 0.7, 'Carbs (g)': 18, 'Fiber (g)': 0.9, 'Vitamin C (mg)': 3.9},
    'Jalepeno': {'Calories': 29, 'Protein (g)': 0.9, 'Carbs (g)': 6.5, 'Fiber (g)': 2.8, 'Vitamin C (mg)': 118.6},
    'Kiwi': {'Calories': 61, 'Protein (g)': 1.1, 'Carbs (g)': 15, 'Fiber (g)': 3.0, 'Vitamin C (mg)': 92.7},
    'Lemon': {'Calories': 29, 'Protein (g)': 1.1, 'Carbs (g)': 9, 'Fiber (g)': 2.8, 'Vitamin C (mg)': 53},
    'Lettuce': {'Calories': 15, 'Protein (g)': 1.4, 'Carbs (g)': 2.9, 'Fiber (g)': 1.3, 'Vitamin C (mg)': 2.8},
    'Mango': {'Calories': 60, 'Protein (g)': 0.8, 'Carbs (g)': 15, 'Fiber (g)': 1.6, 'Vitamin C (mg)': 36.4},
    'Onion': {'Calories': 40, 'Protein (g)': 1.1, 'Carbs (g)': 9, 'Fiber (g)': 1.7, 'Vitamin C (mg)': 7.4},
    'Orange': {'Calories': 47, 'Protein (g)': 0.9, 'Carbs (g)': 12, 'Fiber (g)': 2.4, 'Vitamin C (mg)': 53.2},
    'Paprika': {'Calories': 289, 'Protein (g)': 14.1, 'Carbs (g)': 54, 'Fiber (g)': 34.9, 'Vitamin C (mg)': 144},
    'Pear': {'Calories': 57, 'Protein (g)': 0.4, 'Carbs (g)': 15, 'Fiber (g)': 3.1, 'Vitamin C (mg)': 4.3},
    'Peas': {'Calories': 81, 'Protein (g)': 5.4, 'Carbs (g)': 14, 'Fiber (g)': 5.7, 'Vitamin C (mg)': 40},
    'Pineapple': {'Calories': 50, 'Protein (g)': 0.5, 'Carbs (g)': 13, 'Fiber (g)': 1.4, 'Vitamin C (mg)': 47.8},
    'Pomegranate': {'Calories': 83, 'Protein (g)': 1.7, 'Carbs (g)': 19, 'Fiber (g)': 4, 'Vitamin C (mg)': 10.2},
    'Potato': {'Calories': 77, 'Protein (g)': 2, 'Carbs (g)': 17, 'Fiber (g)': 2.2, 'Vitamin C (mg)': 19.7},
    'Raddish': {'Calories': 16, 'Protein (g)': 0.7, 'Carbs (g)': 3.4, 'Fiber (g)': 1.6, 'Vitamin C (mg)': 14.8},
    'Soy beans': {'Calories': 173, 'Protein (g)': 16.6, 'Carbs (g)': 9.9, 'Fiber (g)': 6.0, 'Vitamin C (mg)': 6},
    'Spinach': {'Calories': 23, 'Protein (g)': 2.9, 'Carbs (g)': 3.6, 'Fiber (g)': 2.2, 'Vitamin C (mg)': 28.1},
    'Sweetcorn': {'Calories': 86, 'Protein (g)': 3.2, 'Carbs (g)': 19, 'Fiber (g)': 2.7, 'Vitamin C (mg)': 6.8},
    'Sweetpotato': {'Calories': 86, 'Protein (g)': 1.6, 'Carbs (g)': 20, 'Fiber (g)': 3, 'Vitamin A (μg)': 709},
    'Tomato': {'Calories': 18, 'Protein (g)': 0.9, 'Carbs (g)': 3.9, 'Fiber (g)': 1.2, 'Vitamin C (mg)': 13.7},
    'Turnip': {'Calories': 28, 'Protein (g)': 0.9, 'Carbs (g)': 6, 'Fiber (g)': 1.8, 'Vitamin C (mg)': 21},
    'Watermelon': {'Calories': 30, 'Protein (g)': 0.6, 'Carbs (g)': 8, 'Fiber (g)': 0.4, 'Vitamin C (mg)': 8.1}
}

scientific_names={
    'Asparagus': 'Asparagus officinalis',
    'Crimson clover': 'Trifolium incarnatum',
    'Daisy fleabane': 'Erigeron annuus',
    'Fireweed': 'Chamerion angustifolium',
    'Milk thistle': 'Silybum marianum',
    'Sunflower': 'Helianthus annuus',
    'Apple': 'Malus domestica',
    'Banana': 'Musa acuminata',
    'Beetroot': 'Beta vulgaris',
    'Bell pepper': 'Capsicum annuum',
    'Cabbage': 'Brassica oleracea',
    'Capsicum': 'Capsicum annuum',
    'Carrot': 'Daucus carota',
    'Cauliflower': 'Brassica oleracea var. botrytis',
    'Chilli pepper': 'Capsicum frutescens',
    'Corn': 'Zea mays',
    'Cucumber': 'Cucumis sativus',
    'Eggplant': 'Solanum melongena',
    'Garlic': 'Allium sativum',
    'Ginger': 'Zingiber officinale',
    'Grapes': 'Vitis vinifera',
    'Jalepeno': 'Capsicum annuum var. jalapeño',
    'Kiwi': 'Actinidia deliciosa',
    'Lemon': 'Citrus limon',
    'Lettuce': 'Lactuca sativa',
    'Mango': 'Mangifera indica',
    'Onion': 'Allium cepa',
    'Orange': 'Citrus sinensis',
    'Paprika': 'Capsicum annuum',
    'Pear': 'Pyrus communis',
    'Peas': 'Pisum sativum',
    'Pineapple': 'Ananas comosus',
    'Pomegranate': 'Punica granatum',
    'Potato': 'Solanum tuberosum',
    'Raddish': 'Raphanus sativus',
    'Soy beans': 'Glycine max',
    'Spinach': 'Spinacia oleracea',
    'Sweetcorn': 'Zea mays var. saccharata',
    'Sweetpotato': 'Ipomoea batatas',
    'Tomato': 'Solanum lycopersicum',
    'Turnip': 'Brassica rapa subsp. rapa',
    'Watermelon': 'Citrullus lanatus'
}

Info={
    "Asparagus": "A nutrient-rich vegetable known for its distinct spear-like shape. High in fiber, vitamins A, C, and K, it's great for overall health. Asparagus also contains antioxidants, which help protect cells from damage. It is a natural diuretic, promoting kidney health and reducing bloating. Asparagus is versatile and can be roasted, steamed, or grilled. It also supports digestive health due to its high fiber content. This vegetable is low in calories and is a great addition to weight-loss diets. Asparagus is also known to improve bone health due to its high vitamin K content.",
    
    "Crimson clover": "A vibrant flowering plant often used as a cover crop. It enhances soil fertility by fixing nitrogen and attracts pollinators. Crimson clover is beneficial for preventing soil erosion due to its fast growth. It also adds organic matter to the soil, improving soil structure. The bright red flowers are not only beautiful but serve as a food source for bees and butterflies. It is often used as a green manure crop to enrich the soil. This hardy plant grows in a variety of soil types and climates. It’s commonly planted in gardens and agricultural fields for its environmental benefits.",
    
    "Daisy fleabane": "A wildflower with daisy-like blooms, often found in meadows. It has traditional uses in herbal medicine for its anti-inflammatory properties. Daisy fleabane attracts pollinators such as bees and butterflies with its sweet nectar. Its seeds provide food for birds, supporting local wildlife. The plant thrives in a variety of soil conditions and climates. Daisy fleabane is also known for its adaptability and ability to grow in disturbed areas. The flowers are often used in floral arrangements due to their simple yet elegant appearance. It’s a great addition to any wildflower garden or natural landscape.",
    
    "Fireweed": "A flowering plant known for its bright pink flowers. It’s commonly used to make tea and has antioxidant properties. Fireweed is a pioneer species, often seen growing in areas affected by fire. Its leaves and flowers are used to create natural remedies for inflammation and digestive issues. The plant is also beneficial for soil stabilization, preventing erosion in disturbed areas. Bees and other pollinators are attracted to its nectar, which is used to produce flavorful honey. Fireweed is also a valuable plant for ecological recovery after wildfires. It has been historically used by Indigenous peoples for medicinal and ceremonial purposes.",
    
    "Milk thistle": "A medicinal herb with purple flowers, valued for liver health. It contains silymarin, a compound that supports detoxification. Milk thistle is also known for its antioxidant and anti-inflammatory effects, which help protect the body from free radical damage. It is commonly used in herbal supplements for its liver-supporting properties. The plant thrives in sunny, well-drained soil and is drought-resistant once established. Milk thistle has been traditionally used to treat liver conditions such as cirrhosis and hepatitis. It may also support healthy cholesterol levels and improve skin health. The seeds of milk thistle are processed into oils and extracts for therapeutic use.",
    
    "Sunflower": "A tall plant with large, bright yellow flowers. Its seeds are a rich source of healthy fats and vitamins. Sunflowers are popular in gardens for their aesthetic appeal and their ability to attract pollinators. The oil extracted from sunflower seeds is commonly used in cooking and skincare products. Sunflowers have deep taproots, making them effective at breaking up compacted soil. They are also used to phytoremediate soil, absorbing heavy metals and toxins. Sunflowers can grow in a variety of climates and are relatively low-maintenance plants. The seeds are nutritious and provide a great snack, as well as being used in baking.",
    
    "Apple": "A popular fruit known for its crisp texture and sweet taste. Packed with fiber and antioxidants, it’s great for heart health. Apples are rich in vitamin C, which supports immune function and skin health. The fruit also contains pectin, a soluble fiber that aids in digestion. Apples come in a variety of colors, including red, green, and yellow, with each variety having its own distinct flavor. They are often consumed raw but are also used in cooking and baking, such as in pies and applesauce. Apples have been linked to improved brain health and may help in reducing the risk of certain chronic diseases. The seeds contain trace amounts of cyanide, which is toxic in large quantities, but generally safe when consumed in moderation.",
    
    "Banana": "A tropical fruit with a creamy texture and sweet taste. It’s a great source of potassium and energy. Bananas are also rich in vitamin B6, which supports metabolism and cognitive function. The high fiber content in bananas promotes gut health and aids digestion. Bananas are versatile, used in smoothies, desserts, and baked goods like banana bread. The fruit is easy to digest, making it ideal for people with digestive issues. They are also a great source of antioxidants, which help combat oxidative stress. Bananas grow in clusters, and their trees can reach up to 30 feet tall in ideal conditions.",
    
    "Beetroot": "A root vegetable with a deep red color. Rich in nitrates, it’s known for boosting stamina and supporting heart health. Beetroot is high in fiber, which aids digestion and helps maintain a healthy gut. It’s also packed with antioxidants, particularly betalains, which reduce inflammation and promote overall health. Beetroot can be eaten raw, roasted, or pickled, and its juice is a popular health drink. The vegetable is often used in salads and smoothies for added nutrition. Beetroot consumption has been linked to improved exercise performance due to enhanced blood flow. It also helps in lowering blood pressure, thanks to its nitrate content.",
    
    "Bell pepper": "A crunchy vegetable available in various colors like green, red, and yellow. High in vitamins A and C, it’s great for immunity. Bell peppers are rich in antioxidants, such as lutein and zeaxanthin, which help protect the eyes. They are also a good source of fiber, supporting digestive health. Bell peppers have a mild, sweet flavor and can be eaten raw in salads or cooked in a variety of dishes. The plant is easy to grow in warm climates and produces fruit in different colors. Bell peppers are low in calories and provide a healthy snack. They are also high in capsaicin, which has been shown to have potential metabolism-boosting benefits.",
    
    "Cabbage": "A leafy vegetable available in green, red, or white varieties. It's rich in fiber and vitamin C, supporting digestion and immunity. Cabbage is known for its high content of antioxidants, which help fight inflammation. It is also a great source of vitamin K, which supports bone health and proper blood clotting. Cabbage can be eaten raw in salads, fermented into sauerkraut, or cooked in stews and soups. The vegetable is low in calories and high in water content, making it ideal for weight management. It is also a natural detoxifier and can aid in liver function. Cabbage is a hearty vegetable, growing well in cooler climates and can be harvested in late fall.",
    
    "Capsicum": "Another name for bell pepper, commonly used in cooking. It adds color, flavor, and nutrition to dishes. Capsicum is rich in vitamin C, which boosts the immune system and helps fight infections. It also contains capsaicin, the compound responsible for the spicy heat in peppers, which has various health benefits. Capsicum is a versatile ingredient, often used in salads, stir-fries, and roasted dishes. It can be eaten raw or cooked, and each color variety has a slightly different flavor profile. It’s also an excellent source of fiber, aiding digestion and promoting gut health. Capsicum plants are easy to grow and thrive in warm, sunny environments.",
    
    "Carrot": "A root vegetable known for its vibrant orange color. High in beta-carotene, it promotes good vision and skin health. Carrots are rich in antioxidants, particularly in their bright orange pigment, which helps protect against oxidative stress. They are also an excellent source of fiber, which supports digestive health. Carrots are low in calories and can be enjoyed raw, cooked, or juiced. The vegetable is versatile and can be added to salads, soups, and stews. Carrots are also high in vitamin K, supporting bone health and proper clotting. They are easy to grow in most climates and are typically harvested in the fall.",
    
    "Cauliflower": "A versatile vegetable with a white, dense head. It's low in calories and high in fiber, making it ideal for healthy diets. Cauliflower is rich in antioxidants, which help reduce inflammation and oxidative stress. It also contains compounds that support detoxification processes in the body. The vegetable can be used as a low-carb substitute for rice or mashed potatoes. Cauliflower is a good source of vitamin C, supporting the immune system and skin health. It can be roasted, steamed, or used in soups and curries. The plant is easy to grow in cooler climates and is typically harvested in the fall.",
    
    "Chilli pepper": "A spicy fruit used to add heat to dishes. Rich in capsaicin, it boosts metabolism and provides pain relief benefits. Chilli peppers are rich in vitamin C, supporting immune function and skin health. Capsaicin has been shown to have anti-inflammatory properties, making it beneficial for joint pain. The peppers can range in heat level from mild to extremely hot, depending on the variety. They are also a good source of antioxidants, particularly carotenoids, which help protect against oxidative stress. Chili peppers are often used fresh, dried, or ground into powders. They are easy to grow in warm, sunny environments.",
    
    "Corn": "A grain often eaten as a vegetable, known for its sweet kernels. It’s a good source of carbohydrates and fiber. Corn is rich in antioxidants, such as lutein and zeaxanthin, which promote eye health. The kernels come in a variety of colors, including yellow, white, and blue, each with its own flavor. Corn is widely used in both whole form and processed into products like cornmeal, corn syrup, and popcorn. It is high in fiber, which supports digestion and regulates blood sugar levels. Corn is commonly grown in warm climates and is a staple food in many cultures. It also has industrial uses, such as in biofuel production.",
    
    "Cucumber": "A hydrating vegetable with high water content. It’s low in calories and great for skin and digestion. Cucumbers are rich in antioxidants, particularly flavonoids and tannins, which help reduce inflammation. They are also a good source of vitamin K, which supports bone health and blood clotting. Cucumbers can be eaten raw in salads or sandwiches, or pickled to make refreshing snacks. The vegetable is low in carbohydrates, making it a great option for low-carb diets. It’s also known for its cooling properties, making it a popular ingredient in skincare products. Cucumbers are easy to grow in gardens, requiring warm temperatures and plenty of sunlight.",
    
    "Eggplant": "A glossy purple vegetable with a meaty texture. It’s rich in antioxidants and versatile in cooking. Eggplant contains nasunin, an antioxidant that protects brain cells from damage. The vegetable is also high in fiber, which supports digestion and gut health. Eggplant can be grilled, roasted, or fried, and is commonly used in dishes like eggplant Parmesan or baba ganoush. It is low in calories and can be a healthy addition to various diets. Eggplant is often used as a meat substitute in vegetarian and vegan cooking due to its hearty texture. It is easy to grow in warm climates and requires plenty of sunlight for optimal growth.",
    
    "Garlic": "A pungent bulb known for its medicinal properties. It boosts immunity and supports heart health. Garlic contains allicin, a compound with antimicrobial and anti-inflammatory effects. It has been used for centuries in traditional medicine for treating infections and digestive issues. Garlic is also known to improve circulation and reduce blood pressure. It can be eaten raw or cooked, and is a staple ingredient in many cuisines worldwide. The vegetable is high in antioxidants, which protect cells from oxidative damage. Garlic is easy to grow in most climates, with a long harvest period.",
    
    "Ginger": "A root spice with a spicy-sweet flavor. It's widely used for its anti-inflammatory and digestive benefits. Ginger is rich in antioxidants, which help reduce oxidative stress in the body. It contains gingerol, a bioactive compound that has been linked to numerous health benefits, including reducing nausea and supporting immune function. Ginger is commonly used in teas, smoothies, and as a spice in cooking. It is also used in traditional medicine for treating digestive issues, such as indigestion and bloating. The root is also believed to have anti-cancer properties, although more research is needed. Ginger can be easily grown in warm, humid climates and harvested in 10-12 months.",
    
    "Grapes": "Small, juicy fruits available in red, green, or black. They’re rich in antioxidants and good for heart health. Grapes are high in vitamins C and K, supporting immune function and bone health. They are also a source of fiber, promoting digestive health. Grapes are commonly eaten raw, dried into raisins, or processed into juice and wine. They contain flavonoids, which have been shown to reduce the risk of heart disease by improving blood vessel function. Grapes are also a great snack, offering a quick energy boost. They grow in clusters on vines, typically in warm, dry climates.",
    
    "Jalapeno": "A medium-sized chili pepper known for its mild to moderate heat. It's often used to spice up dishes. Jalapenos are rich in vitamin C, promoting healthy skin and immune function. They also contain capsaicin, which has been linked to metabolism-boosting and pain-relief benefits. The pepper can be eaten fresh, pickled, or dried, and is commonly used in Mexican and Tex-Mex dishes. Jalapenos have a distinct flavor, offering both heat and a mild sweetness. They are often sliced into salads, salsas, or added to sandwiches for an extra kick. Jalapenos are easy to grow in warm, sunny conditions.",
    
    "Kiwi": "A small fruit with green flesh and tiny seeds. High in vitamin C, it supports immunity and skin health. Kiwi is also rich in antioxidants, particularly vitamin E, which protects cells from damage. The fruit contains fiber, which aids digestion and supports gut health. Kiwi can be eaten raw by scooping out the flesh with a spoon or added to fruit salads and smoothies. It is low in calories and provides a natural energy boost. Kiwi also contains actinidin, an enzyme that helps break down protein and improve digestion. This tropical fruit thrives in mild climates and is easy to grow with proper care.",
    
    "Lemon": "A citrus fruit known for its tangy flavor. Rich in vitamin C, it aids digestion and boosts immunity. Lemons are also high in antioxidants, which help protect the body from oxidative damage. The fruit is commonly used in cooking, beverages, and as a garnish. Lemon juice is often used to make refreshing lemonades or to enhance the flavor of dishes. Lemons are known for their cleansing properties, promoting detoxification and supporting liver function. The peel contains essential oils that can be used for cleaning and aromatherapy. Lemons grow on trees and thrive in sunny, temperate climates.",
    
    "Lettuce": "A leafy green vegetable often used in salads. Low in calories and high in water content, it’s refreshing and healthy. Lettuce is a good source of vitamin K, which supports bone health. It also contains folate, which is essential for cell division and overall health. Lettuce comes in several varieties, including romaine, iceberg, and butterhead, each with unique textures and flavors. It can be eaten raw, sautéed, or added to wraps and sandwiches. Lettuce is a staple in many diets due to its light flavor and nutritional value. It is easy to grow in cool climates and is harvested in early spring and fall.",
    
    "Mango": "A tropical fruit with sweet, juicy flesh. It's rich in vitamin A and antioxidants, great for skin and vision. Mangoes are also a source of fiber, which aids digestion and promotes gut health. The fruit contains several vitamins, including vitamins C and E, which support immune function and skin health. Mangoes come in different varieties, with some being sweeter and others having a more tart flavor. They are commonly eaten fresh, in smoothies, or used in salads and desserts. Mangoes are easy to grow in warm, tropical climates and are a popular fruit around the world.",
    
    "Onion": "A staple vegetable with a pungent flavor. It’s used in various cuisines and has antimicrobial and antioxidant properties. Onions are rich in sulfur compounds, which have been shown to support heart health and reduce the risk of cancer. The vegetable is a good source of vitamin C, which boosts immunity and skin health. Onions can be eaten raw, sautéed, roasted, or added to soups, stews, and salads. They also contain prebiotics, which promote healthy gut bacteria and digestive health. Onions can be grown in a variety of soils and climates, with different types ranging from sweet to pungent varieties.",
    
    "Orange": "A citrus fruit with sweet and tangy flavor. Packed with vitamin C, it’s excellent for immunity and skin health. Oranges are also rich in fiber, supporting digestion and gut health. The fruit contains flavonoids, which have antioxidant and anti-inflammatory properties. Oranges can be eaten raw, juiced, or used in a variety of dishes and desserts. The peel is also used to make zest, which adds flavor to recipes. Oranges are low in calories and provide a natural energy boost. They grow on trees and thrive in warm, sunny climates, typically harvested in the winter months.",
    
    "Paprika": "A ground spice made from dried peppers. It adds vibrant color and mild heat to dishes. Paprika is a good source of vitamin A, which supports vision and skin health. The spice contains antioxidants, such as carotenoids, which help protect the body from oxidative damage. Paprika is commonly used in Hungarian and Spanish cuisine, adding a distinctive flavor to dishes like goulash and paella. It can be mild or spicy, depending on the type of pepper used. Paprika also has antimicrobial properties and is sometimes used as a preservative. It is easy to store and can last for months in a cool, dry place.",
    
    "Pear": "A sweet and juicy fruit with a fibrous core. It's rich in dietary fiber and antioxidants, supporting digestion. Pears are high in vitamin C, which supports immune health and skin vitality. The fruit has a smooth texture and a slightly grainy interior, making it unique among fruits. Pears can be eaten raw, baked, or used in jams and salads. They are also known for their hydrating properties, as they have a high water content. Pears come in various varieties, from green to yellow and red, each with its own flavor and texture. The fruit grows on trees and thrives in temperate climates.",
    
    "Peas": "Small green seeds that grow in pods. High in protein and fiber, they’re a staple in many diets. Peas are rich in vitamins A, C, and K, which support immune function and bone health. They are also a good source of antioxidants, particularly flavonoids, which help reduce inflammation. Peas can be eaten fresh, frozen, or dried, and are commonly used in soups, stews, and salads. They are low in calories but provide a good amount of protein, making them a great choice for vegetarians and vegans. Peas grow in cool weather and can be harvested in spring or fall.",
    
    "Pineapple": "A tropical fruit with a tangy-sweet flavor. It’s rich in vitamin C and bromelain, an enzyme that aids digestion. Pineapple is also high in manganese, which supports bone health and metabolism. The fruit contains antioxidants, such as flavonoids, that help protect the body from oxidative damage. Pineapple can be eaten fresh, grilled, or used in a variety of dishes, including smoothies and desserts. It is commonly used in tropical cuisine, including fruit salads and salsas. Pineapple grows in tropical climates and requires well-drained soil and plenty of sunlight.",
    
    "Pomegranate": "A fruit with juicy red seeds called arils. High in antioxidants, it’s great for heart health and immunity. Pomegranates are also a good source of vitamin C, which supports immune function and skin health. The fruit contains compounds called punicalagins, which have been shown to have anti-inflammatory and anti-cancer properties. Pomegranates can be eaten fresh, juiced, or used in cooking and baking. The seeds are rich in fiber, which promotes digestive health. Pomegranates grow on small shrubs or trees and thrive in warm, sunny climates.",
    
    "Potato": "A starchy vegetable that’s a staple food worldwide. It’s versatile in cooking and a good source of energy. Potatoes are high in potassium, which supports heart and muscle function. They also contain vitamin C, which boosts immunity and skin health. Potatoes can be boiled, mashed, roasted, or fried, and are commonly used in dishes like mashed potatoes and fries. The vegetable is low in fat but high in carbohydrates, making it a great source of energy. Potatoes grow in cool climates and can be stored for months in a cool, dry place.",
    
    "Raddish": "A crunchy root vegetable with a peppery taste. It’s low in calories and aids digestion. Radishes are high in vitamin C, which supports immune function and skin health. The vegetable is also rich in antioxidants, particularly anthocyanins, which have been linked to improved heart health. Radishes are often eaten raw in salads or sandwiches, but can also be roasted, grilled, or pickled. The vegetable is rich in fiber, which supports digestive health and regularity. Radishes grow quickly and are easy to cultivate in cool climates, often harvested in early spring or fall.",
    
    "Soy beans": "A legume high in protein and a key ingredient in many plant-based foods. It’s great for muscle building and heart health. Soybeans are rich in essential amino acids, which are important for tissue growth and repair. They contain phytoestrogens, which may have benefits for hormonal balance and bone health. Soybeans can be eaten whole, in the form of tofu or tempeh, or used to make soy milk and oil. The legume is a good source of fiber, which supports digestive health. Soybeans are commonly grown in temperate climates and require well-drained soil for optimal growth.",
    
    "Spinach": "A leafy green rich in iron and vitamins A and C. It’s excellent for bone health and immunity. Spinach is also a good source of magnesium, which supports muscle and nerve function. The vegetable is rich in antioxidants, including lutein, which supports eye health. Spinach can be eaten raw in salads or cooked in a variety of dishes, such as soups, stews, and smoothies. It is low in calories, making it ideal for weight loss diets. Spinach is easy to grow in cool climates and requires well-drained soil and plenty of sunlight.",
    
    "Sweetcorn": "The sweet variety of corn often eaten on the cob. It’s high in fiber and natural sugars. Sweetcorn is also rich in antioxidants, such as lutein and zeaxanthin, which support eye health. The kernels are a good source of vitamin C, which supports immunity and skin health. Sweetcorn can be boiled, grilled, or roasted, and is often used in salads, soups, and casseroles. It is high in carbohydrates, which provide a quick energy source. Sweetcorn is commonly grown in warm climates and is a popular side dish at barbecues.",
    
    "Sweetpotato": "A root vegetable with orange flesh. Rich in beta-carotene, it’s great for skin and eye health. Sweetpotatoes are also high in fiber, which supports digestion and promotes gut health. The vegetable contains vitamin C, which boosts immune function and skin health. Sweetpotatoes can be baked, mashed, roasted, or used in soups and casseroles. They are often used in both savory and sweet dishes, such as pies and casseroles. Sweetpotatoes are easy to grow in warm climates and require plenty of sunlight for optimal growth.",
    
    "Tomato": "A juicy fruit often used as a vegetable in cooking. It’s rich in lycopene, an antioxidant that supports heart health. Tomatoes are also a good source of vitamin C, which supports immune function and skin health. The fruit contains potassium, which supports muscle and nerve function. Tomatoes can be eaten raw in salads or sandwiches, or cooked in sauces, soups, and stews. They are low in calories and rich in flavor. Tomatoes thrive in warm climates and are commonly grown in home gardens.",
    
    "Turnip": "A root vegetable with a slightly peppery flavor. It’s low in calories and high in vitamins C and K. Turnips are also a good source of fiber, which supports digestion and gut health. The vegetable can be eaten raw, roasted, boiled, or mashed. Turnips are often used in soups, stews, and casseroles, and can also be pickled for a tangy snack. The vegetable contains antioxidants that help reduce inflammation and support heart health. Turnips grow well in cool climates and can be harvested in the fall.",
    
    "Watermelon": "A hydrating fruit with sweet, juicy flesh. It’s rich in vitamins A and C and great for summer refreshment. Watermelon is also a good source of lycopene, an antioxidant that promotes heart health. The fruit contains citrulline, an amino acid that helps improve blood flow. Watermelon is low in calories and high in water, making it perfect for hydration during hot weather. It can be eaten fresh, blended into smoothies, or used in fruit salads. Watermelon is easy to grow in warm climates and requires well-drained soil and plenty of sunlight." 
}




# Category lists
fruits = ['Apple', 'Banana', 'Bell pepper', 'Chilli pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Asparagus', 'Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger', 'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy beans', 'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip']
medicinal_plants = ['Crimson clover', 'Daisy fleabane', 'Fireweed', 'Garlic', 'Milk thistle', 'Sunflower']



def fetch_calories(prediction):
    try:
        # Set a user-agent header to simulate a real browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        url = f'https://www.google.com/search?&q=calories+in+{prediction}'
        req = requests.get(url, headers=headers).text
        scrap = BeautifulSoup(req, 'html.parser')

        # Locate the calories information using the inner class
        calories = scrap.find("div", class_="Z0LcW an_fna")

        if calories:
            return calories.text.strip()  # Return the calories text with no leading or trailing spaces
        else:
            st.error("Specific Calories not provided")
            return None
    except Exception as e:
        st.error("Can't fetch the Calories")
        print(e)



def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


# def run():
#     st.set_page_config(page_title="Ingredient Classifier", page_icon="🍍")

#     # Custom CSS for aesthetics
#     st.markdown("""
#         <style>
#         .main {
#             background-color: #f4f4f4;
#             font-family: Arial, sans-serif;
#         }
#         .stButton>button {
#             color: white;
#             background-color: #4CAF50;
#             padding: 10px 24px;
#             border: none;
#             border-radius: 8px;
#             cursor: pointer;
#         }
#         .stButton>button:hover {
#             background-color: #45a049;
#         }
#         .card {
#             background-color: white;
#             padding: 15px;
#             border-radius: 10px;
#             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
#         }
#         </style>
#     """, unsafe_allow_html=True)
#     st.title("Ingredients🍍🍅 Classification")
#     img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
#     if img_file is not None:
#         img = Image.open(img_file).resize((250, 250))
#         st.image(img, use_column_width=False)
#         save_image_path = './upload_images/' + img_file.name
#         with open(save_image_path, "wb") as f:
#             f.write(img_file.getbuffer())

#         # if st.button("Predict"):
#         if img_file is not None:
#             result = processed_img(save_image_path)
#             print(result)
#             print(scientific_names[result])
#             st.success("**Original Name : " + result + '**')
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.info(f"**Scientific Name**: {scientific_names[result]}")
#             with col2:
#                 category = "Fruit" if result in fruits else "Vegetable" if result in vegetables else "Medicinal Plant"
#                 st.info(f"**Category**: {category}")
#             st.markdown(f"<div class='card'><p>{Info[result]}</p></div>", unsafe_allow_html=True)
            
#             # cal = fetch_calories(result)
#             # if cal:
#             #     st.warning('**' + cal + ' (100 grams)**')
#             st.title("Nutritional Values of "+result)

#             ingredient = result
#             if ingredient:
#                 if ingredient in nutritional_values:
#                     ingredient_df = pd.DataFrame.from_dict(
#                         {ingredient: nutritional_values[ingredient]}, orient='index'
#                     )

#                     # Display the table
#                     st.table(ingredient_df)
#                 else:
#                     st.error(f"No nutritional information found for '{ingredient}'.")


# run()


def run():
    # App layout
    st.set_page_config(page_title="Ingredient Classifier", layout="wide")
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 3rem;
            text-align: center;
            color: #2a9d8f;
            font-family: 'Arial Black', sans-serif;
        }
        .sidebar {
            background-color: #f4a261;
        }
        .success-title {
            font-size: 1.5rem;
            color: #264653;
        }
        .info-text {
            color: #e76f51;
        }
        .card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main title
    st.markdown('<h1 class="main-title">🍍 Ingredient Classifier 🍅</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Upload Your Image 📂")
        img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
        st.markdown(
            "### Instructions 📜\n"
            "1. Upload an image of an ingredient.\n"
            "2. Click **Predict** to see the results.\n"
            "3. View detailed nutritional values and scientific info.",
            unsafe_allow_html=True,
        )

    if img_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(img_file), width=300, caption="Uploaded Image")

        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Provide a unique key for the "Predict" button
        if st.button("Predict", key="predict_button_1"):
            progress_bar = st.progress(0)  # Initialize progress bar
            progress_text = st.empty()
                
            # Simulate processing and updating the progress bar
            for i in range(100):
                progress_bar.progress(i + 1)
                progress_text.text(f"Analyzing the image... {i + 1}%")
                time.sleep(0.05)  # Simulating image analysis time
            st.markdown('</div>', unsafe_allow_html=True)

            # Process the image and get the prediction result
            result = processed_img(save_image_path)
            
            if result:  # Check if result is valid
                st.markdown(f"<h2 class='success-title'>Predicted Ingredient: {result}</h2>", unsafe_allow_html=True)
                st.info(f"**Scientific Name:** {scientific_names.get(result, 'N/A')}")
                
                category = (
                    "Vegetable" if result in vegetables
                    else "Medicinal Plant" if result in medicinal_plants
                    else "Fruit"
                )
                st.warning(f"**Category:** {category}")
                st.markdown(f"<div class='card'><p>{Info.get(result, 'Information not available')}</p></div>", unsafe_allow_html=True)

                st.markdown(f"<h3 class='success-title'>Nutritional Values of {result}</h3>", unsafe_allow_html=True)
                
                if result in nutritional_values:
                    ingredient_df = pd.DataFrame.from_dict(
                        {result: nutritional_values[result]}, orient='index'
                    )
                    st.table(ingredient_df)
                else:
                    st.error(f"No nutritional information found for '{result}'.")
            else:
                st.error("Failed to predict the ingredient. Please try again.")

run()
