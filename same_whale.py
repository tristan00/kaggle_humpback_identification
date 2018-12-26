#TODO:


image_size = 224


def get_net2():
    base_model1 = MobileNetV2(include_top=False, input_shape=(image_size, image_size, 3))
    base_model2 = MobileNetV2(include_top=False, input_shape=(image_size, image_size, 3))

    x = Flatten()(base_model1.output)
    x2 = Flatten()(base_model2.output)
    x_conc = Concatenetate(x, x2)
    x3 = Dense(2096)
    x4 = Dense(2096)
    x5 - Dense(1)

    model = Model(input = base_model.input, output = x)
    adam = optimizers.Adam(lr=.0001, decay=1e-5)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    return model