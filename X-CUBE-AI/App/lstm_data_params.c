/**
  ******************************************************************************
  * @file    lstm_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Thu Oct  6 20:18:38 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "lstm_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_lstm_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_lstm_weights_array_u64[301] = {
  0xbc131cbabe9ac204U, 0xbb9b2d323cdfbf45U, 0x3e3c4d9c3ec2d19bU, 0xbe2248c9be97bd9aU,
  0x3dda29853e246551U, 0xbe82751ebcda55b4U, 0x3da8c0d03e4a165cU, 0xbe5f9bfcbe6b6b0aU,
  0x3d7c44993e145618U, 0xbd21d1c8beae079fU, 0xbbb905d3bc9235a1U, 0xbea21d853e9d1d41U,
  0x3da8d385be68af4cU, 0xbeda3ccb3ced67aaU, 0xbed6c1ed3e688102U, 0xbe5c70f3be7afa01U,
  0xbdb44836bc446d51U, 0x3ed3fe4cbe254637U, 0xbe8a5ac6bb685c5bU, 0x3d507713be253a1cU,
  0xbe6fd16d3e8a46ffU, 0xbdc4a5cfbe7d0f26U, 0x3eaf6a7abe708c82U, 0x3df508d43f022585U,
  0x3eb650e5be699f93U, 0x3e540e5bbdeea7a1U, 0xbd5793553e6e5f7bU, 0x3dfd74c83d2316b6U,
  0x3ec948313e603377U, 0x3ee83395bd97e1d2U, 0x3e13d6b03ec682eeU, 0xbe8944a4be7c3746U,
  0xbe1287bf3e9e44ecU, 0x3e54a68c3eb8fd33U, 0xbea10f513ec1c4ebU, 0xbcde9186bc32b9f8U,
  0x3d96d7bc3e94296dU, 0x3e4fbd4d3e7975f1U, 0x3e8082c7bebb62b2U, 0xbe29933a3e334710U,
  0xbe62917ebe79b69aU, 0xbe87e1de3dbe1a83U, 0xbcb54a5a3e8500edU, 0x3d64a915be9c125bU,
  0xbd8161233de2287cU, 0xbe90a5d1be4b5879U, 0xbe5f82f9bea8768fU, 0xbd2339713db4c761U,
  0xbe9143363e6dbd95U, 0x3e95ba793dd75882U, 0x3e16b784be8230fbU, 0xbe441b35bd481853U,
  0x3da34d38be62b29fU, 0xbd20a6f13e8ce14dU, 0xbdf2081d3ea61802U, 0xbdb17bfcbeb270eeU,
  0x3d95309c3d9867caU, 0xbd4d17803ea94a6bU, 0x3c9310e93e6a0432U, 0xbe6e18bf3e7d0fb7U,
  0xbdd766c6be883a9eU, 0x3ebbeb90bd7e797dU, 0x3d12ac333e13ad77U, 0x3f11c32ebf094486U,
  0x3e933c3e3ce10c20U, 0xbda775c33dbe386aU, 0xbdc29c1d3e42b7b5U, 0xbed996b2bce07e53U,
  0xbe65058abe808da8U, 0xbe65b1913b0cd63eU, 0xbdff629c3e91d225U, 0xbd110b79bf0ae6fcU,
  0x3bdaf615bd060a85U, 0x3ee664a8bba99eedU, 0xbceef1f13e23640fU, 0xbba8fd2abf7d5eceU,
  0x3dacb88fbedf99b1U, 0x3c5860a8bf407339U, 0xbe0eed3d3e68b2dfU, 0xbd63d7a03e235d9dU,
  0x3dbf9e5e3d72364dU, 0x3de8cbb53dbb9c28U, 0xbe942d70bd92be1cU, 0xbeefd1d63c30bfd8U,
  0x3e4acd82bdc03a9fU, 0x3d661d25bc04de3eU, 0x3e02bc61be5b879aU, 0xbe53ab13be07f583U,
  0x3ccbe46fbd41e7f0U, 0xbe65a380bebc9fc3U, 0xbdd8a543bc8d505dU, 0xbe88fa04be47823cU,
  0xbf35873a3d0363a6U, 0xbc10c458bdfddb08U, 0xbe150c11be77b68fU, 0xbc2b7cda3f872a75U,
  0x3e5fc3223edc712cU, 0x3b8e42a43f1696dbU, 0xbe270d893e51e200U, 0xbe9ca5f6bd26b4b2U,
  0xbd517f543e6ac449U, 0xbe6a43ec3e7338e1U, 0x3d6b8a3b3e86ded8U, 0xbe963a03be6ecfc6U,
  0x3e382f5abe4677d1U, 0x3d02f4893e18ec20U, 0x3e6468843e1edfbbU, 0xbf05d6ad3e440b50U,
  0xbe18ee8a3e19cb87U, 0xbd45dbf33d685bdbU, 0x3eb1349ebdf1884cU, 0xbe006d61bdd6c491U,
  0xbe8a5b84bea6b4bbU, 0x3e3531243eb1b432U, 0xbe9227343db8be20U, 0xbe16423abee0e727U,
  0xbcb7a1483e9d4be4U, 0xbe880d6c3e634774U, 0x3e0049043d8585bdU, 0xbe9367153cf80916U,
  0xbe4627acbee0b817U, 0xbf857d2ebde56a8eU, 0xbf07a5da3c7cae86U, 0xbeb876873d6dce13U,
  0xbe752a2f3e4ab6d6U, 0xbe5e6b6bbe595498U, 0xbe3ca383bf792f51U, 0x3bd688fdbf48fb73U,
  0xbe92c9e2be81fdf7U, 0x3ce94b8a3e1eab17U, 0xbe7d3c19be04664bU, 0x3f80d5ccbe1bc8b2U,
  0xbe2ff4ddbe26d088U, 0x3ea6092d3cbc10caU, 0xbe0b7c8fbdad236eU, 0xbe0a7a253e9bbdd3U,
  0xbddeb4b1bdb58d9dU, 0xbed22da63cd4f13fU, 0xbe4a817dbb1070c2U, 0xbe86a956be6be284U,
  0xbc4bd64bbeee978bU, 0x3c1b22793d80d809U, 0xbe8fbbc23d61039bU, 0x3df371a7bd97540dU,
  0xbc2dde2abdea34e2U, 0xbd1f8da2bd891f0bU, 0xbdc164c73f35d416U, 0xbeac0e6e3ed18451U,
  0x3c78436d3c362376U, 0xbbd360943e242c82U, 0xbde626113ec9660bU, 0xbe82b37f3e5b7136U,
  0x3d4495a2bd0bf722U, 0x3d850353bd8c584bU, 0xbd88e1823d57bc8bU, 0xbe4d46403dec625fU,
  0xbe8c22bdbd3f167dU, 0xbe992b483dbde128U, 0xbf00720cbe8b38faU, 0xbeeb1941bda56f9fU,
  0x3dc54385bd1bf0a8U, 0xbe0bd691bd935e1fU, 0x3e1b134e3d387799U, 0x3eb01962be445145U,
  0xbe7aee05bbb49c48U, 0xbe159fc3bd86af40U, 0x3bab22b1be184a98U, 0xbd8c225fbdb2df0dU,
  0xbd8bf8103e09e6b5U, 0x3e2d96c33baaa66aU, 0xbe99150cbe98ad85U, 0xbec46c40bdebb909U,
  0x3e82b5633eedf230U, 0x3e04b489bcefbde5U, 0xbd09bf023e1a37cdU, 0xbdc665e83e26de6aU,
  0x3db7df903cc8a8a9U, 0xbe18283abe328c42U, 0xbda9dabfbe75b9b3U, 0xbb5cc952bc918af8U,
  0xbe9058e53e06ce82U, 0x3e3ad2773e4bba7bU, 0x3d65a243be7e6934U, 0xbd030e3fbe46eb09U,
  0x3cd006de3c168783U, 0x3e9cb35abd2b9726U, 0x3dd4aaf6be47ca64U, 0xbdc13d16be0f8910U,
  0xbd6ed90d3cfd8b57U, 0xbe977d923eccae03U, 0xbe424de63e1a5437U, 0xbce7db783e417da4U,
  0x3daf9783bd46cce7U, 0xbe89a9c8bd624577U, 0xbe1b7177be813b50U, 0xbdf353d2bd38d858U,
  0xbe39ce853b839e07U, 0x3d95c435bcf60774U, 0x3e692a18be0c44a4U, 0xbd4aaa383dc63b38U,
  0x3e2734d5be591ac4U, 0xbd1c7f3b3e819944U, 0x3e1bd17fbed9bf61U, 0xbe6a0ab4bd863d85U,
  0x3e39ddadbe5b13a4U, 0x3e6f603c3e14f540U, 0x3e91c93c3d9e5f71U, 0xbe65655ebd3608d1U,
  0x3e4f87c1bebf1eb0U, 0xbeb41ae93e82bc2aU, 0xbeb8a21fbdad99fdU, 0x3df7f849be35b3f7U,
  0x3d6718e03ec51b88U, 0x3f0b7715bedbb86eU, 0x3e063555beccb9edU, 0xbe093e20bf36cbd1U,
  0x3e592360bbc65934U, 0xbf0306313db4ee18U, 0x3e82101cbe3fe875U, 0xbea17afb3d061e26U,
  0xbe6020e13d985b8dU, 0x3e3032c7bd4c9dbbU, 0x3d8576ebbe093292U, 0x3f09b535bcbb96adU,
  0x3dd0f2fe3d2f597eU, 0xbe658609bf162692U, 0xbe80a392bf24b9c8U, 0x3e97a995bf49e9adU,
  0xbe0aea683c2b902aU, 0xbdb215aabe8f4681U, 0xbda940a73d29d292U, 0xbeb3c1bcbe86c3fdU,
  0xbd98e9a4be109920U, 0xbf178f2d3cea236bU, 0x3e15c4f53d6acda0U, 0x3dbbbc09be0219c3U,
  0x3e9ea49dbdfd12c9U, 0xbe9c8c40bdb582f1U, 0xbe377a72be8d6505U, 0xbe405fdb3db83b5fU,
  0x3e0be8f5be54707cU, 0xbe0386febd9b8669U, 0xbec043773ca0add8U, 0x3e18bd7cbdf7c5faU,
  0xbec2f23abdf76abeU, 0xbe58a0b93e6a085eU, 0x3bb0c4723f0e6056U, 0xbcddc8213f1baaffU,
  0x3c5081fc3d4bd143U, 0xbe1c8c3b3e8ec4e2U, 0xbef415d93e469970U, 0xbe0c6ee3bdda5fedU,
  0x3eb2edb0be58a5d0U, 0xbee79494bec0fa30U, 0x3db5b8c0be443473U, 0x3dd01941bd12f212U,
  0x3e74482ebdf14ad8U, 0xbde634cdbe012356U, 0x3e7088f53e002b34U, 0xbeca95ac3cc275c2U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0x0U,
  0x0U, 0x0U, 0x0U, 0xbdb63b03bbfc6125U,
  0xbade434ebb88696bU, 0xbd3b4c63bdcd5eebU, 0xbd346c69bda49669U, 0xbde2e38fbddbbdbdU,
  0x3f5e99bb3f61d67dU, 0x3f7f6bbd3f781f2cU, 0x3f56918a3f618d4eU, 0x3f7085a03f7109b7U,
  0x3f3d2bb03f3e7f22U, 0xbca62f073d5df9b9U, 0xbc38add43c6e2275U, 0x3daf54253c97ea9eU,
  0xbc07cac7bd2da6b2U, 0x3d3c32e0bc47e625U, 0xbdea6fccbbcbe433U, 0xbcbb5892bbaa5730U,
  0xbd476e82bdd41332U, 0xbd36f987bd679cf7U, 0xbddbd33abde2000dU, 0x3f196950be5db391U,
  0xbf2b4762be18af03U, 0x3db6e6083eb0425aU, 0x3ef1a80f3d6b98d9U, 0x3d1d74b1bd07ef0bU,
  0x3d09b0edU,
};


ai_handle g_lstm_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_lstm_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

